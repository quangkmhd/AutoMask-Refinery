import os
import typer
from typing import Optional
from pathlib import Path

from automask_refinery.config.settings import settings
from automask_refinery.core.detector import MaskDetector
from automask_refinery.core.generator import MaskGenerator
from automask_refinery.utils.organizer import FileOrganizer
from automask_refinery.utils.logger import log

app = typer.Typer(help="AutoMask-Refinery: Professional Mask Quality Control Toolkit")

@app.command()
def generate(
    data_dir: str = typer.Option(settings.DATA_DIR, help="Path to input images directory"),
    sam_model: str = typer.Option(os.path.join(settings.PROJECT_ROOT, "sam3.pt"), help="Path to SAM3 model"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-generation of existing masks"),
    imgsz: int = typer.Option(1036, help="SAM3 input image size")
):
    """Generate masks using SAM3."""
    log.info(f"Generating masks in {data_dir}...")
    generator = MaskGenerator(sam_model)
    generator.generate_for_directory(data_dir, force=force, imgsz=imgsz)

@app.command()
def detect(
    data_dir: str = typer.Option(settings.DATA_DIR, help="Path to dataset directory"),
    review_out: str = typer.Option(settings.REVIEW_OUT, help="Output for failed samples"),
    passed_out: str = typer.Option(settings.PASSED_OUT, help="Output for passed samples"),
    limit: int = typer.Option(200, help="Limit number of visualizations")
):
    """Detect failed masks using statistical and heuristic methods."""
    log.info(f"Detecting failed masks in {data_dir}...")
    detector = MaskDetector(data_dir)
    dataset = detector.load_dataset()
    if not dataset:
        log.error("No dataset loaded. Check paths.")
        return

    results, final_failed = detector.run_pipeline(dataset)
    passed_samples = [s for s in results if not s['final_failed']]
    
    detector.visualize_results(final_failed, review_out, limit=limit)
    detector.visualize_results(passed_samples, passed_out, limit=limit)
    
    log.info("Detection complete.")

@app.command()
def ui(
    port: int = typer.Option(settings.PORT, help="Port to run the UI"),
    host: str = typer.Option(settings.HOST, help="Host to run the UI"),
    data_dir: str = typer.Option(settings.DATA_DIR, help="Path to data directory")
):
    """Launch the interactive review dashboard."""
    from automask_refinery.app import run_app
    settings.PORT = port
    settings.HOST = host
    settings.DATA_DIR = data_dir
    run_app()

@app.command()
def organize(
    source: str = typer.Option(settings.DATA_DIR, help="Source root containing folders"),
    pass_dir: str = typer.Option(settings.PASSED_OUT, help="Destination for passed files"),
    fail_dir: str = typer.Option(settings.REVIEW_OUT, help="Destination for failed files"),
    csv: str = typer.Option(settings.SUMMARY_CSV, help="Path to review_details.csv"),
    move: bool = typer.Option(False, "--move", help="Move files instead of copying")
):
    """Organize files into pass/fail folders based on review results."""
    organizer = FileOrganizer(source, pass_dir, fail_dir, csv)
    organizer.organize(move=move)

if __name__ == "__main__":
    app()
