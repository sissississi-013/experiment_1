import time
from pathlib import Path
from validation_pipeline.config import PipelineConfig
from validation_pipeline.errors import DatasetError, SpecValidationError, PipelineError
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.report import FinalReport
import validation_pipeline.modules.dataset_resolver as _dataset_resolver_mod
from validation_pipeline.modules.spec_generator import generate_spec
from validation_pipeline.modules.calibrator import calibrate
from validation_pipeline.modules.planner import generate_plan
from validation_pipeline.modules.compiler import compile_plan
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.modules.supervisor import supervise
from validation_pipeline.modules.reporter import generate_report
from validation_pipeline.tools.registry import ToolRegistry
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import (
    ModuleStarted, ModuleCompleted, SpecGenerated, PlanGenerated,
    DatasetResolved, PipelineErrorEvent,
)


class ValidationPipeline:
    def __init__(self, config: PipelineConfig | None = None, event_bus: EventBus | None = None):
        self.config = config or PipelineConfig()
        self.event_bus = event_bus or EventBus()
        self.registry = ToolRegistry(self.config.tool_configs_dir)

    def run(self, user_input: UserInput, auto_approve: bool = False) -> FinalReport:
        # Module 0: Dataset Resolution (conditional)
        if not user_input.dataset_path or not Path(user_input.dataset_path).exists():
            if user_input.dataset_description:
                self.event_bus.publish(ModuleStarted(module="dataset_resolver"))
                t = time.time()
                try:
                    dataset_plan = _dataset_resolver_mod.resolve_dataset(user_input.dataset_description, self.config)
                    local_path = _dataset_resolver_mod.download_dataset(dataset_plan)
                    user_input = user_input.model_copy(update={"dataset_path": local_path})
                except PipelineError as e:
                    self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
                    raise
                self.event_bus.publish(ModuleCompleted(module="dataset_resolver", duration_seconds=time.time() - t))
                self.event_bus.publish(DatasetResolved(module="dataset_resolver", source=dataset_plan.source, image_count=len(list(Path(local_path).iterdir())), download_path=local_path))
            elif not user_input.dataset_path:
                raise DatasetError(
                    "Either dataset_path or dataset_description must be provided",
                    module="pipeline",
                )

        # Module 1: Spec Generator
        self.event_bus.publish(ModuleStarted(module="spec_generator"))
        t = time.time()
        try:
            spec = generate_spec(user_input, self.config)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="spec_generator", duration_seconds=time.time() - t))
        self.event_bus.publish(SpecGenerated(module="spec_generator", spec_summary=spec.restated_request, quality_criteria=[qc.dimension for qc in spec.quality_criteria], content_criteria=[cc.object_or_scene for cc in spec.content_criteria]))

        if auto_approve:
            spec.user_confirmed = True
        if not spec.user_confirmed:
            raise SpecValidationError(
                "Spec must be confirmed by user before proceeding",
                module="pipeline",
            )

        # Module 2: Calibrator
        self.event_bus.publish(ModuleStarted(module="calibrator"))
        t = time.time()
        try:
            dim_to_tool = {}
            for qc in spec.quality_criteria:
                matches = self.registry.search_by_task("image_quality")
                if matches:
                    tool = self.registry.get_tool(matches[0]["name"])
                    dim_to_tool[qc.dimension] = tool

            cal_result = calibrate(
                spec,
                user_input.exemplar_good_paths,
                user_input.exemplar_bad_paths,
                dim_to_tool,
            )
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="calibrator", duration_seconds=time.time() - t))

        # Module 3: Planner (only show tools that are actually available)
        self.event_bus.publish(ModuleStarted(module="planner"))
        t = time.time()
        try:
            available_tool_configs = [
                t_cfg for t_cfg in self.registry.list_tools()
                if self.registry.configs.get(t_cfg["name"]) is not None
            ]
            plan = generate_plan(spec, cal_result, available_tool_configs, self.config)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="planner", duration_seconds=time.time() - t))
        self.event_bus.publish(PlanGenerated(module="planner", steps_count=len(plan.steps), tiers=sorted(set(s.tier for s in plan.steps))))

        if auto_approve:
            plan.user_approved = True

        # Filter plan steps to only use tools we can actually load
        loadable = set()
        for t_cfg in available_tool_configs:
            try:
                self.registry.get_tool(t_cfg["name"])
                loadable.add(t_cfg["name"])
            except Exception:
                pass
        plan.steps = [s for s in plan.steps if s.tool_name in loadable]

        if not plan.user_approved:
            raise SpecValidationError(
                "Plan must be approved by user before proceeding",
                module="pipeline",
            )

        # Module 4: Compiler
        self.event_bus.publish(ModuleStarted(module="compiler"))
        t = time.time()
        try:
            program = compile_plan(plan)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="compiler", duration_seconds=time.time() - t))

        # Module 5: Executor
        self.event_bus.publish(ModuleStarted(module="executor"))
        t = time.time()
        try:
            tools = {}
            unavailable = []
            for tool_name in program.tool_imports:
                try:
                    tools[tool_name] = self.registry.get_tool(tool_name)
                except KeyError:
                    unavailable.append(tool_name)
            if unavailable:
                import sys
                print(f"WARNING: Tools not available (skipping): {unavailable}", file=sys.stderr)

            result = execute_program(program, user_input.dataset_path, tools, cal_result, event_bus=self.event_bus)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="executor", duration_seconds=time.time() - t))

        # Module 6: Supervisor
        self.event_bus.publish(ModuleStarted(module="supervisor"))
        t = time.time()
        try:
            supervision = supervise(result, cal_result, plan)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="supervisor", duration_seconds=time.time() - t))

        # Module 7: Reporter
        self.event_bus.publish(ModuleStarted(module="reporter"))
        t = time.time()
        try:
            report = generate_report(result, supervision, spec, plan)
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
            raise
        self.event_bus.publish(ModuleCompleted(module="reporter", duration_seconds=time.time() - t))

        return report
