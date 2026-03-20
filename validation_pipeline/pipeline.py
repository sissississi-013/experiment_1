from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.report import FinalReport
from validation_pipeline.modules.spec_generator import generate_spec
from validation_pipeline.modules.calibrator import calibrate
from validation_pipeline.modules.planner import generate_plan
from validation_pipeline.modules.compiler import compile_plan
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.modules.supervisor import supervise
from validation_pipeline.modules.reporter import generate_report
from validation_pipeline.tools.registry import ToolRegistry


class ValidationPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.registry = ToolRegistry(self.config.tool_configs_dir)

    def run(self, user_input: UserInput, auto_approve: bool = False) -> FinalReport:
        # Module 1: Spec Generator
        spec = generate_spec(user_input, self.config)
        if auto_approve:
            spec.user_confirmed = True
        if not spec.user_confirmed:
            raise ValueError("Spec must be confirmed by user before proceeding")

        # Module 2: Calibrator
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

        # Module 3: Planner
        plan = generate_plan(spec, cal_result, self.registry.list_tools(), self.config)
        if auto_approve:
            plan.user_approved = True
        if not plan.user_approved:
            raise ValueError("Plan must be approved by user before proceeding")

        # Module 4: Compiler
        program = compile_plan(plan)

        # Module 5: Executor
        tools = {}
        for tool_name in program.tool_imports:
            try:
                tools[tool_name] = self.registry.get_tool(tool_name)
            except KeyError:
                pass

        result = execute_program(program, user_input.dataset_path, tools, cal_result)

        # Module 6: Supervisor
        supervision = supervise(result, cal_result, plan)

        # Module 7: Reporter
        report = generate_report(result, supervision, spec, plan)

        return report
