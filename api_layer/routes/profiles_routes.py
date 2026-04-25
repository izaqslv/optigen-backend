from fastapi import APIRouter, HTTPException, Query, Depends, Path
import datetime
import os
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from data_layer.sedimentation.dataset_loader import try_load_dataset
from business_layer.sedimentation.profiles import (
    list_available_fluids,
    list_heights_for_fluid,
    get_profile_timeseries,
)
from business_layer.sedimentation.plotter import (
    generate_profile_plot_from_dataset,
    format_metadata_text
)

from api_layer.security.dependencies import get_current_user
from pydantic import BaseModel
from business_layer.sedimentation.model_inference import predict_concentration, predict_curve, compare_with_experimental
from business_layer.sedimentation.plotter import generate_curve_plot, generate_comparison_plot


router = APIRouter(
    prefix="/profiles",
    tags=["profiles"]
)

#-----------------------------------------------------------------------------------------------------------------------
@router.get(
    "/available_fluids",
    tags=["profiles"],
    summary="List available fluids",
    description="Returns a sorted list of all fluids available in the JSON dataset.",
)
async def list_fluids(user: str = Depends(get_current_user)):
    try:
        dataset = try_load_dataset()
        fluids = list_available_fluids(dataset)
        return {
            "success": True,
            "n_fluids": len(fluids),
            "fluids": fluids,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------------------------------------------------------------------------
@router.get(
    "/available_heights",
    tags=["profiles"],
    summary="List available heights for a fluid",
    description="Returns all measurement heights (cm) available for a given fluid.",
)
async def list_heights(user: str = Depends(get_current_user), fluid_id: int= Query(..., description="Fluid ID (5–10)", example=7)):
    try:
        dataset = try_load_dataset()
        heights = list_heights_for_fluid(dataset, fluid_id)
        return {
            "success": True,
            "fluid_id": fluid_id,
            "n_heights": len(heights),
            "heights_cm": heights,
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------------------------------------------------------------------------
@router.get(
    "/{fluid_id}/height/timeseries",
    tags=["profiles"],
    summary="Get concentration timeseries",
    description="Return concentration timeseries for a given fluid_id and height (cm).",
)
async def profiles_timeseries(user: str = Depends(get_current_user),
    fluid_id: int= Path(..., description="Fluid ID (5–10)", example=9),
    height: float= Query(..., description="Choose a height (cm)", example=12),
    show_metadata: bool = Query(True, description="Include metadata in payload."),
):
    dataset = try_load_dataset()
    try:
        ts = get_profile_timeseries(
            dataset,
            fluid_id,
            height,
            show_metadata=show_metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = {
        "success": True,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "payload": {
            "fluid_id": ts["fluid_id"],
            "height": ts["height"],
            "tempo": ts["tempo"],
            "concentracao": ts["concentracao"],
        },
    }
    if show_metadata:
        result["payload"]["metadata"] = ts.get("metadata", {})
    return result

#-----------------------------------------------------------------------------------------------------------------------
@router.get(
    "/{fluid_id}/height/plot",
    tags=["profiles"],
    summary="Plot profile (PNG + optional save)",
    description="Generate PNG plot (time vs concentration). Use save=true to persist file on server.",
)
async def profiles_plot(user: str = Depends(get_current_user),
    fluid_id: int= Path(..., description="Fluid ID (5–10)", example=6),
    height: float= Query(..., description="Choose a height (cm)", example=18),
    save: bool = Query(
        False,
        description="If true, saves PNG to storage/plots and returns the path.",
    ),
    show_metadata: bool = Query(
        True,
        description="If true, includes fluid metadata in response.",
    ),
):
    dataset = try_load_dataset()
    save_path = None

    # Define output directory if saving
    png_outdir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage")),
        "plots",
    )
    if save:
        os.makedirs(png_outdir, exist_ok=True)
        fname = (
            f"fluid{fluid_id}_h{height}_profile_{int(datetime.datetime.utcnow().timestamp())}.png"
            .replace(".", "_")
        )
        save_path = os.path.join(png_outdir, fname)

    try:
        png_bytes, metadata = generate_profile_plot_from_dataset(
            dataset,
            fluid_id,
            height,
            return_png=True,
            save_path=save_path,
            show_metadata=show_metadata,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Optionally save file (if generate_profile_plot_from_dataset didn't already)
    if save and save_path and png_bytes:
        with open(save_path, "wb") as f:
            f.write(png_bytes)

    img_b64 = base64.b64encode(png_bytes).decode("utf-8") if png_bytes else None

    return {
        "success": True,
        "fluid_id": fluid_id,
        "height": height,
        "saved_path": save_path,
        "img_base64": img_b64,
        "metadata": metadata if show_metadata else None,
    }

@router.get("/{fluid_id}/plot_all", tags=["profiles"])
def plot_all_profiles(fluid_id: int= Path(..., description="Fluid ID (5–10)", example=6), show_metadata: bool = True, user: str = Depends(get_current_user)):
    try:
        dataset = try_load_dataset()

        # 🔹 alturas direto da função
        heights = list_heights_for_fluid(dataset, fluid_id)

        if not heights:
            return {"error": "Nenhuma altura encontrada"}

        plt.figure(figsize=(12, 7))

        colors = plt.cm.viridis(np.linspace(0, 1, len(heights)))

        metadata = None

        for i, h in enumerate(sorted(heights)):

            ts = get_profile_timeseries(
                dataset,
                fluid_id,
                h,
                show_metadata=True
            )

            time = ts["tempo"]
            conc = ts["concentracao"]

            # 🔹 LINHA (modelo)
            plt.plot(
                time,
                conc,
                color=colors[i],
                linewidth=2,
                # label=f"{h} cm"
            )

            plt.plot(
                time,
                conc,
                color=colors[i],
                linewidth=2,
                marker='o',
                markersize=4,
                label=f"{h} cm"
            )

            if metadata is None:
                metadata = ts.get("metadata")

        # 🔹 estética
        plt.xlabel("Tempo (dias)")
        plt.ylabel("Concentração (fração)")
        plt.title(f"Fluido {fluid_id} - Perfis Experimentais")
        plt.grid(True, linestyle='--', alpha=0.4)

        plt.legend(
            fontsize=9,
            ncol=2,
            loc='upper left',
            bbox_to_anchor=(1.02, 1)
        )

        # 🔹 metadata
        if show_metadata and metadata:
            text = format_metadata_text(metadata)
            plt.text(
                1.02,
                0.5,
                text,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.6)
            )

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close("all")

        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "img_base64": img_base64,
            "metadata": metadata
        }

    except Exception as e:
        return {"error": str(e)}
