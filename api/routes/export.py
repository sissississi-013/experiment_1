import io
import json
import zipfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["export"])

@router.get("/api/runs/{run_id}/export")
async def export_run(run_id: str, request: Request, filter: str = "usable", include_manifest: bool = True, include_report: bool = False):
    store = request.app.state.store
    if not store:
        raise HTTPException(500, "No database configured")
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    if filter == "usable":
        verdicts = ["usable"]
    elif filter == "usable+recoverable":
        verdicts = ["usable", "recoverable"]
    else:
        verdicts = None

    images = store.get_run_images(run_id)
    if verdicts:
        images = [img for img in images if img.get("verdict") in verdicts]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            img_path = Path(img.get("image_path", ""))
            if img_path.exists():
                zf.write(img_path, img_path.name)
        if include_manifest:
            manifest = {
                "run_id": run_id,
                "intent": run.get("intent", ""),
                "filter": filter,
                "image_count": len(images),
                "images": [{"image_id": img.get("image_id"), "filename": Path(img.get("image_path", "")).name, "verdict": img.get("verdict"), "scores": img.get("scores", {})} for img in images],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        if include_report:
            report_json = run.get("report_json")
            if report_json:
                zf.writestr("report.json", report_json if isinstance(report_json, str) else json.dumps(report_json, indent=2))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=validation-{run_id}-{filter}.zip"})
