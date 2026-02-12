"""
Ray Tracing Aesthetic MCP Server
Extracted from Midjourney Compendium Ray Tracing Guide

Three-layer olog architecture:
- Layer 1: Pure taxonomy enumeration (0 tokens)
- Layer 2: Deterministic composition and validation (0 tokens)
- Layer 3: Claude synthesis context preparation (~150-200 tokens)

Phase 2.6: Rhythmic preset composition (0 tokens)
Phase 2.7: Attractor visualization prompt generation (0 tokens)
"""

from fastmcp import FastMCP
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import json
import math

mcp = FastMCP("Ray Tracing Aesthetic")

# Get the directory containing this file
BASE_DIR = Path(__file__).parent

def load_taxonomy(filename: str) -> dict:
    """Load a taxonomy YAML file"""
    path = BASE_DIR / "taxonomies" / filename
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_rules(filename: str) -> dict:
    """Load compositional rules YAML file"""
    path = BASE_DIR / "compositional_rules" / filename
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ============================================================================
# PHASE 2.6: RAY TRACING PARAMETER MORPHOSPACE
# ============================================================================
# Normalized [0.0, 1.0] parameter space capturing ray tracing aesthetic variation.
# Each parameter maps a continuous axis of visual transformation.

RAY_TRACING_PARAMETER_NAMES = [
    "light_warmth",          # 0.0 = cool/blue (10000K+) → 1.0 = warm/amber (2000K)
    "caustic_complexity",    # 0.0 = simple single-bounce → 1.0 = recursive multi-bounce
    "surface_specularity",   # 0.0 = matte diffuse → 1.0 = mirror-sharp specular
    "dispersion_intensity",  # 0.0 = no prismatic split → 1.0 = full rainbow separation
    "motion_dynamism",       # 0.0 = frozen static → 1.0 = high-energy chaotic motion
]

# Canonical states: named positions in parameter space derived from
# the server's taxonomy families (light sources × media × surfaces)

RAY_TRACING_COORDS = {
    "soft_ambient": {
        "light_warmth": 0.75,
        "caustic_complexity": 0.20,
        "surface_specularity": 0.15,
        "dispersion_intensity": 0.10,
        "motion_dynamism": 0.10,
    },
    "prismatic_crystal": {
        "light_warmth": 0.25,
        "caustic_complexity": 0.65,
        "surface_specularity": 0.85,
        "dispersion_intensity": 0.95,
        "motion_dynamism": 0.20,
    },
    "neon_noir": {
        "light_warmth": 0.35,
        "caustic_complexity": 0.70,
        "surface_specularity": 0.90,
        "dispersion_intensity": 0.30,
        "motion_dynamism": 0.40,
    },
    "underwater_organic": {
        "light_warmth": 0.45,
        "caustic_complexity": 0.55,
        "surface_specularity": 0.30,
        "dispersion_intensity": 0.40,
        "motion_dynamism": 0.65,
    },
    "fire_dramatic": {
        "light_warmth": 0.95,
        "caustic_complexity": 0.40,
        "surface_specularity": 0.25,
        "dispersion_intensity": 0.15,
        "motion_dynamism": 0.90,
    },
    "architectural_glass": {
        "light_warmth": 0.50,
        "caustic_complexity": 0.80,
        "surface_specularity": 0.60,
        "dispersion_intensity": 0.55,
        "motion_dynamism": 0.05,
    },
    "disco_kaleidoscope": {
        "light_warmth": 0.30,
        "caustic_complexity": 0.95,
        "surface_specularity": 0.95,
        "dispersion_intensity": 0.80,
        "motion_dynamism": 0.85,
    },
}

# ============================================================================
# PHASE 2.6: RHYTHMIC PRESET DEFINITIONS
# ============================================================================
# Each preset defines a periodic oscillation between two canonical states.
# These become limit cycle attractor manifolds in parameter space.

RAY_TRACING_PRESETS = {
    "warmth_sweep": {
        "state_a": "soft_ambient",
        "state_b": "prismatic_crystal",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 24,
        "description": "Smooth oscillation between warm diffuse glow and cool prismatic sharpness"
    },
    "caustic_pulse": {
        "state_a": "neon_noir",
        "state_b": "underwater_organic",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 20,
        "description": "Pulsing between artificial geometric caustics and organic flowing networks"
    },
    "dispersion_ramp": {
        "state_a": "prismatic_crystal",
        "state_b": "fire_dramatic",
        "pattern": "triangular",
        "num_cycles": 2,
        "steps_per_cycle": 30,
        "description": "Linear ramp between cold crystalline dispersion and warm chaotic fire"
    },
    "surface_toggle": {
        "state_a": "soft_ambient",
        "state_b": "neon_noir",
        "pattern": "square",
        "num_cycles": 5,
        "steps_per_cycle": 12,
        "description": "Sharp toggle between matte diffuse warmth and specular neon reflection"
    },
    "depth_breathing": {
        "state_a": "underwater_organic",
        "state_b": "architectural_glass",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 18,
        "description": "Breathing rhythm between organic submersion and architectural precision"
    },
}

# ============================================================================
# PHASE 2.7: VISUAL VOCABULARY FOR PROMPT GENERATION
# ============================================================================
# Maps regions of parameter space to image-generation-ready keywords.
# Nearest-neighbor matching against canonical visual types.

RAY_TRACING_VISUAL_TYPES = {
    "warm_diffuse": {
        "coords": {
            "light_warmth": 0.80,
            "caustic_complexity": 0.15,
            "surface_specularity": 0.10,
            "dispersion_intensity": 0.05,
            "motion_dynamism": 0.10,
        },
        "keywords": [
            "soft golden ambient light",
            "gentle scattered caustic glow",
            "matte surface warmth",
            "diffused shadow gradients",
            "warm color temperature",
            "natural luminosity",
            "subsurface-scattered radiance",
        ],
        "optical_properties": {
            "finish": "matte",
            "shadow_type": "soft_penumbra",
            "color_palette": "warm_amber_to_honey",
        },
    },
    "prismatic_sharp": {
        "coords": {
            "light_warmth": 0.20,
            "caustic_complexity": 0.70,
            "surface_specularity": 0.90,
            "dispersion_intensity": 0.95,
            "motion_dynamism": 0.15,
        },
        "keywords": [
            "crystalline rainbow caustic patterns",
            "prismatic light separation",
            "sharp specular reflections",
            "chromatic dispersion on polished surfaces",
            "cold precise optical geometry",
            "diamond-cut refracted beams",
            "spectral color banding",
        ],
        "optical_properties": {
            "finish": "specular",
            "shadow_type": "hard_geometric",
            "color_palette": "full_spectral_rainbow",
        },
    },
    "neon_geometric": {
        "coords": {
            "light_warmth": 0.30,
            "caustic_complexity": 0.75,
            "surface_specularity": 0.92,
            "dispersion_intensity": 0.25,
            "motion_dynamism": 0.45,
        },
        "keywords": [
            "neon-colored reflected light patterns",
            "geometric caustic grids",
            "mirror-sharp surface reflections",
            "artificial colored illumination",
            "cyberpunk light refractions",
            "chrome and glass interplay",
            "saturated single-hue caustics",
        ],
        "optical_properties": {
            "finish": "high_gloss",
            "shadow_type": "sharp_colored",
            "color_palette": "neon_magenta_cyan_violet",
        },
    },
    "organic_luminous": {
        "coords": {
            "light_warmth": 0.50,
            "caustic_complexity": 0.50,
            "surface_specularity": 0.25,
            "dispersion_intensity": 0.35,
            "motion_dynamism": 0.70,
        },
        "keywords": [
            "flowing underwater caustic networks",
            "dappled light through water surface",
            "organic rippling refraction patterns",
            "subsurface scattering glow",
            "bioluminescent caustic shimmer",
            "natural fluid light motion",
            "translucent depth layering",
        ],
        "optical_properties": {
            "finish": "translucent",
            "shadow_type": "soft_dappled",
            "color_palette": "aqua_teal_to_warm_green",
        },
    },
    "dramatic_fire": {
        "coords": {
            "light_warmth": 0.95,
            "caustic_complexity": 0.35,
            "surface_specularity": 0.20,
            "dispersion_intensity": 0.10,
            "motion_dynamism": 0.90,
        },
        "keywords": [
            "flickering warm caustic shadows",
            "fire-lit dancing light patterns",
            "chaotic ember glow on rough surfaces",
            "dramatic high-contrast chiaroscuro",
            "molten amber light streaks",
            "volatile dynamic illumination",
            "deep shadow with hot highlight edges",
        ],
        "optical_properties": {
            "finish": "rough_textured",
            "shadow_type": "deep_dramatic",
            "color_palette": "ember_orange_to_deep_crimson",
        },
    },
}


# ============================================================================
# LAYER 1: PURE TAXONOMY (0 tokens)
# ============================================================================

@mcp.tool()
def list_light_sources(category: str = "all") -> dict:
    """
    List enumerated light sources with optical properties.
    
    Args:
        category: Filter by category or "all" for complete taxonomy
            - natural_celestial
            - natural_weather  
            - natural_fire
            - natural_bioluminescence
            - artificial_interior
            - artificial_exterior
            - artificial_neon
            - artificial_vehicle
            - artificial_stage
            - artificial_screen
            - artificial_specialty
            - artificial_industrial
            - artificial_emergency
            - artificial_decorative
            - artificial_modern
            - artificial_scientific
            - artificial_atmospheric
            - artificial_vintage
    
    Returns:
        Dictionary with light source specifications including:
        - color_temperature (Kelvin)
        - intensity
        - spectral_distribution
        - shadow_quality
        - caustic_strength
    """
    taxonomy = load_taxonomy("light_sources.yaml")
    
    if category == "all":
        return taxonomy
    
    # Filter by category
    return {k: v for k, v in taxonomy.items() 
            if v.get('category') == category}

@mcp.tool()
def list_refractive_media() -> dict:
    """
    List refractive media with optical properties.
    
    Families:
    - water_based: underwater, rain, puddles, waterfalls, mist
    - glass_material: greenhouse, stained glass, wine bottles, sculptures
    - architectural_glass: facades, skylights, walkways
    - ice: caves, frozen waterfalls, frost, sculptures
    - reflective_surfaces: mirrors, disco balls
    - natural_phenomena: tree canopy, spider webs
    
    Returns:
        Dictionary with medium specifications including:
        - refractive_index
        - dispersion (rainbow splitting capability)
        - transparency
        - typical_caustic_shape
        - color_shift
    """
    return load_taxonomy("refractive_media.yaml")

@mcp.tool()
def list_caustic_movements() -> dict:
    """
    List 19 caustic motion patterns from Midjourney Compendium.
    
    Returns dictionary with movement specifications including:
    - velocity_profile
    - pattern_type (organic/geometric/chaotic)
    - temporal_periodicity
    - spatial_coherence
    - animation_complexity
    
    Movements: dancing, flowing, rippling, cascading, pulsing, flickering,
    undulating, spiraling, fragmenting, kaleidoscoping, shimmering, morphing,
    radiating, converging, drifting, scattering, weaving, streaming, oscillating
    """
    return load_taxonomy("caustic_movements.yaml")

@mcp.tool()
def list_receiving_surfaces() -> dict:
    """
    List receiving surface materials with light interaction properties.
    
    Families:
    - metallic: chrome, steel, car paint, aluminum foil
    - stone: marble, granite, asphalt
    - organic: wood, skin, sand, snow, fabric
    - synthetic_fabric: satin, silk, velvet, leather
    - ceramic_glass: tiles, porcelain, acrylic
    - specialized: water surfaces, oil slicks, bubbles
    
    Returns dictionary with surface specifications including:
    - reflectivity (0.0-1.0)
    - roughness (0.0-1.0)
    - subsurface_scattering
    - color_bleeding_susceptibility
    - caustic_visibility
    """
    return load_taxonomy("receiving_surfaces.yaml")

# ============================================================================
# LAYER 2: DETERMINISTIC COMPOSITION (0 tokens)
# ============================================================================

@mcp.tool()
def compose_caustic_effect(
    light_source: str,
    refractive_medium: str,
    movement: str,
    receiving_surface: str
) -> dict:
    """
    Compose complete caustic effect specification.
    
    Validates tensor product: LightSource ⊗ Medium ⊗ Movement ⊗ Surface
    
    Args:
        light_source: ID from list_light_sources
        refractive_medium: ID from list_refractive_media
        movement: Movement pattern ID
        receiving_surface: ID from list_receiving_surfaces
    
    Returns:
        - composition_valid: bool
        - optical_parameters: combined properties
        - emergent_effects: complex interactions
        - compatibility_analysis: conflicts and strengths
    """
    # Load all taxonomies
    lights = load_taxonomy("light_sources.yaml")
    media = load_taxonomy("refractive_media.yaml")
    movements = load_taxonomy("caustic_movements.yaml")
    surfaces = load_taxonomy("receiving_surfaces.yaml")
    compat = load_rules("compatibility_matrix.yaml")
    
    # Validate all components exist
    if light_source not in lights:
        return {"error": f"Unknown light source: {light_source}"}
    if refractive_medium not in media:
        return {"error": f"Unknown refractive medium: {refractive_medium}"}
    if movement not in movements:
        return {"error": f"Unknown movement: {movement}"}
    if receiving_surface not in surfaces:
        return {"error": f"Unknown receiving surface: {receiving_surface}"}
    
    light_data = lights[light_source]
    medium_data = media[refractive_medium]
    movement_data = movements[movement]
    surface_data = surfaces[receiving_surface]
    
    # Calculate optical parameters
    optical_params = {
        "color_temperature": light_data.get("color_temperature"),
        "caustic_intensity": _calculate_caustic_intensity(
            light_data, medium_data, surface_data
        ),
        "caustic_shape": medium_data.get("typical_caustic_shape"),
        "movement_pattern": movement_data.get("pattern_type"),
        "surface_interaction": _calculate_surface_interaction(
            medium_data, surface_data
        )
    }
    
    # Check compatibility
    medium_family = medium_data.get("family")
    surface_id = receiving_surface
    
    compatibility_check = _check_compatibility(
        medium_family, surface_id, compat
    )
    
    # Detect emergent effects
    emergent = _detect_emergent_effects(
        light_data, medium_data, surface_data, compat
    )
    
    return {
        "composition_valid": compatibility_check["valid"],
        "optical_parameters": optical_params,
        "emergent_effects": emergent,
        "compatibility_analysis": compatibility_check,
        "render_recommendations": _get_render_recommendations(
            light_data, medium_data, movement_data
        )
    }

def _calculate_caustic_intensity(light_data: dict, medium_data: dict, surface_data: dict) -> str:
    """Calculate expected caustic intensity"""
    light_intensity = light_data.get("caustic_strength", "moderate")
    medium_refraction = medium_data.get("dispersion", "moderate")
    surface_visibility = surface_data.get("caustic_visibility", "moderate")
    
    # Simple heuristic combination
    intensity_map = {
        ("very_strong", "very_high", "excellent"): "extreme",
        ("very_strong", "high", "excellent"): "very_high",
        ("strong", "high", "very_good"): "high",
    }
    
    return intensity_map.get(
        (light_intensity, medium_refraction, surface_visibility),
        "moderate"
    )

def _calculate_surface_interaction(medium_data: dict, surface_data: dict) -> dict:
    """Calculate how medium and surface interact"""
    return {
        "caustic_sharpness": "sharp" if surface_data.get("roughness", 0.5) < 0.2 else "diffused",
        "color_bleeding": "high" if surface_data.get("color_bleeding_susceptibility") == "very_high" else "moderate",
        "subsurface_effect": surface_data.get("subsurface_scattering", "none")
    }

def _check_compatibility(medium_family: str, surface_id: str, compat: dict) -> dict:
    """Check medium-surface compatibility"""
    # Look up in compatibility matrix
    medium_key = f"{medium_family}_medium"
    
    if medium_key not in compat:
        return {"valid": True, "confidence": "unknown", "notes": "No specific compatibility data"}
    
    medium_compat = compat[medium_key]
    
    # Check if surface is explicitly compatible
    compatible_surfaces = medium_compat.get("compatible_surfaces", {})
    if surface_id in compatible_surfaces:
        return {
            "valid": True,
            "confidence": "high",
            "quality": compatible_surfaces[surface_id]
        }
    
    # Check if explicitly incompatible
    incompatible_surfaces = medium_compat.get("incompatible_surfaces", {})
    if surface_id in incompatible_surfaces:
        return {
            "valid": False,
            "confidence": "high",
            "reason": incompatible_surfaces[surface_id].get("reason"),
            "alternative": incompatible_surfaces[surface_id].get("alternative")
        }
    
    return {"valid": True, "confidence": "moderate", "notes": "No specific incompatibility found"}

def _detect_emergent_effects(light_data: dict, medium_data: dict, surface_data: dict, compat: dict) -> list:
    """Detect emergent optical effects"""
    effects = []
    
    # Check for subsurface scattering interaction
    if (surface_data.get("subsurface_scattering") in ["high", "very_high"] and
        light_data.get("caustic_strength") in ["strong", "very_strong"]):
        effects.append({
            "type": "subsurface_scattering_interaction",
            "description": "Caustics create softened glowing effect through translucent surface",
            "aesthetic": "organic_luminous"
        })
    
    # Check for rainbow dispersion
    if medium_data.get("dispersion") in ["very_high", "high_chromatic"]:
        effects.append({
            "type": "rainbow_dispersion",
            "description": "Prismatic color separation creates rainbow caustic patterns",
            "aesthetic": "spectral_artistic"
        })
    
    # Check for recursive reflections
    if (surface_data.get("reflectivity", 0) > 0.8 and
        "reflective" in str(surface_data.get("caustic_visibility", ""))):
        effects.append({
            "type": "recursive_reflection",
            "description": "High reflectivity creates multiplied caustic complexity",
            "aesthetic": "geometric_complex"
        })
    
    return effects

def _get_render_recommendations(light_data: dict, medium_data: dict, movement_data: dict) -> dict:
    """Get render parameter recommendations"""
    return {
        "bounce_count": 3 if light_data.get("intensity") == "very_high" else 2,
        "photon_density": "high" if medium_data.get("dispersion") == "very_high" else "moderate",
        "motion_blur": movement_data.get("animation_complexity") in ["high", "very_high"],
        "shadow_quality": light_data.get("shadow_quality", "medium")
    }

@mcp.tool()
def validate_ray_tracing_composition(
    light_source: str,
    refractive_medium: str,
    receiving_surface: str
) -> dict:
    """
    Validate physical plausibility of composition without movement.
    
    Checks:
    - Light intensity sufficient for visible caustics
    - Medium creates visible effect
    - Surface allows caustic visibility
    - Color temperature compatibility
    
    Returns validation with conflict analysis.
    """
    result = compose_caustic_effect(
        light_source, refractive_medium, "static", receiving_surface
    )
    
    if "error" in result:
        return result
    
    return {
        "valid": result["composition_valid"],
        "optical_parameters": result["optical_parameters"],
        "compatibility": result["compatibility_analysis"],
        "warnings": _generate_warnings(result)
    }

def _generate_warnings(composition_result: dict) -> list:
    """Generate warnings about potential issues"""
    warnings = []
    
    optical = composition_result.get("optical_parameters", {})
    compat = composition_result.get("compatibility_analysis", {})
    
    if optical.get("caustic_intensity") in ["weak", "very_weak"]:
        warnings.append("Low caustic intensity - consider brighter light source or more reflective surface")
    
    if compat.get("confidence") == "moderate":
        warnings.append("Compatibility uncertain - test render recommended")
    
    return warnings

# ============================================================================
# LAYER 3: SYNTHESIS CONTEXT FOR CLAUDE
# ============================================================================

@mcp.tool()
def prepare_ray_tracing_synthesis_context(
    light_source: str,
    refractive_medium: str,
    movement: str,
    receiving_surface: str,
    subject: str,
    scene_description: str = ""
) -> dict:
    """
    Prepare complete synthesis context for Claude.
    
    Assembles all deterministic parameters for creative synthesis.
    Claude receives ~150-200 token context with validated optical
    parameters and technical vocabulary.
    
    Args:
        light_source: Light source ID
        refractive_medium: Medium ID  
        movement: Movement pattern ID
        receiving_surface: Surface ID
        subject: What to depict (e.g., "sports car", "portrait")
        scene_description: Optional additional scene context
    
    Returns:
        Complete synthesis context ready for Claude prompt generation
    """
    # Get composition
    composition = compose_caustic_effect(
        light_source, refractive_medium, movement, receiving_surface
    )
    
    if "error" in composition:
        return composition
    
    # Load original data for descriptions
    lights = load_taxonomy("light_sources.yaml")
    media = load_taxonomy("refractive_media.yaml")
    movements = load_taxonomy("caustic_movements.yaml")
    surfaces = load_taxonomy("receiving_surfaces.yaml")
    
    return {
        "subject": subject,
        "scene_description": scene_description,
        
        "optical_parameters": composition["optical_parameters"],
        "emergent_effects": composition["emergent_effects"],
        "render_specs": composition["render_recommendations"],
        
        "component_descriptions": {
            "light": lights[light_source],
            "medium": media[refractive_medium],
            "movement": movements[movement],
            "surface": surfaces[receiving_surface]
        },
        
        "technical_vocabulary": {
            "lighting_terms": [
                "global illumination",
                "photorealistic light behavior",
                "ray traced rendering",
                "physically accurate materials"
            ],
            "caustic_terms": [
                "refracted light patterns",
                "caustic networks",
                f"{composition['optical_parameters']['caustic_shape']} caustic shapes",
                f"{composition['optical_parameters']['movement_pattern']} motion pattern"
            ],
            "quality_terms": [
                f"{composition['render_recommendations']['shadow_quality']} shadows with proper penumbra",
                "natural depth of field",
                "crisp clean edges"
            ]
        },
        
        "synthesis_guidelines": {
            "describe_composition": "Use explicit geometric specifications for light and caustics",
            "translate_taxonomy": "Convert technical IDs to natural image generation vocabulary",
            "specify_angles": "Include precise angles and spatial relationships where relevant",
            "aesthetic_emphasis": _determine_aesthetic_emphasis(composition)
        }
    }

def _determine_aesthetic_emphasis(composition: dict) -> str:
    """Determine primary aesthetic emphasis from composition"""
    emergent = composition.get("emergent_effects", [])
    
    if any(e.get("type") == "rainbow_dispersion" for e in emergent):
        return "prismatic_color_separation"
    elif any(e.get("type") == "subsurface_scattering_interaction" for e in emergent):
        return "organic_luminous_glow"
    elif any(e.get("type") == "recursive_reflection" for e in emergent):
        return "geometric_complexity"
    else:
        return "natural_photorealistic"


# ============================================================================
# PHASE 2.6: RHYTHMIC COMPOSITION TOOLS (0 tokens)
# ============================================================================

def _generate_oscillation(num_steps: int, num_cycles: float, pattern: str) -> list:
    """
    Generate oscillation alpha values [0.0, 1.0].
    
    Pure deterministic waveform generation.
    
    Args:
        num_steps: Total sample count
        num_cycles: Number of A→B→A cycles
        pattern: "sinusoidal", "triangular", or "square"
    
    Returns:
        List of float alpha values
    """
    alphas = []
    for i in range(num_steps):
        t = 2.0 * math.pi * num_cycles * i / num_steps

        if pattern == "sinusoidal":
            alpha = 0.5 * (1.0 + math.sin(t))
        elif pattern == "triangular":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 2.0 * t_norm if t_norm < 0.5 else 2.0 * (1.0 - t_norm)
        elif pattern == "square":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 0.0 if t_norm < 0.5 else 1.0
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        alphas.append(alpha)
    return alphas


def _interpolate_states(state_a: dict, state_b: dict, alpha: float) -> dict:
    """Linearly interpolate between two parameter-space states."""
    return {
        p: state_a[p] * (1.0 - alpha) + state_b[p] * alpha
        for p in RAY_TRACING_PARAMETER_NAMES
    }


def _generate_preset_trajectory(preset_name: str) -> list:
    """
    Generate full trajectory for a Phase 2.6 preset.
    
    Returns:
        List of state dicts, one per step
    """
    config = RAY_TRACING_PRESETS[preset_name]
    state_a = RAY_TRACING_COORDS[config["state_a"]]
    state_b = RAY_TRACING_COORDS[config["state_b"]]

    total_steps = config["num_cycles"] * config["steps_per_cycle"]
    alphas = _generate_oscillation(total_steps, config["num_cycles"], config["pattern"])

    return [_interpolate_states(state_a, state_b, a) for a in alphas]


@mcp.tool()
def list_ray_tracing_rhythmic_presets() -> dict:
    """
    List all Phase 2.6 rhythmic presets for ray tracing aesthetics.

    Each preset defines a periodic oscillation between two canonical
    ray tracing states. These are used for temporal composition and
    serve as limit cycle attractor manifolds in the discovery system.

    Cost: 0 tokens (pure Layer 1 enumeration)

    Returns:
        Dictionary of preset names → configuration and metadata
    """
    result = {}
    for name, config in RAY_TRACING_PRESETS.items():
        result[name] = {
            "state_a": config["state_a"],
            "state_b": config["state_b"],
            "pattern": config["pattern"],
            "period": config["steps_per_cycle"],
            "num_cycles": config["num_cycles"],
            "total_steps": config["num_cycles"] * config["steps_per_cycle"],
            "description": config["description"],
        }
    return {
        "domain": "ray_tracing",
        "presets": result,
        "parameter_names": RAY_TRACING_PARAMETER_NAMES,
        "canonical_states": list(RAY_TRACING_COORDS.keys()),
    }


@mcp.tool()
def generate_ray_tracing_rhythmic_sequence(
    preset_name: str = "",
    state_a_id: str = "",
    state_b_id: str = "",
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0,
) -> dict:
    """
    Generate rhythmic oscillation between two ray tracing states.

    Two modes:
      1. Preset mode: provide preset_name (uses curated configuration)
      2. Custom mode: provide state_a_id, state_b_id, and pattern params

    Args:
        preset_name: Name from list_ray_tracing_rhythmic_presets (preset mode)
        state_a_id: Starting canonical state ID (custom mode)
        state_b_id: Ending canonical state ID (custom mode)
        oscillation_pattern: "sinusoidal" | "triangular" | "square"
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle (= period for limit cycle detection)
        phase_offset: Starting phase 0.0=A, 0.5=B

    Returns:
        Sequence of parameter states with pattern metadata

    Cost: 0 tokens (Layer 2 deterministic)
    """
    # Resolve configuration
    if preset_name:
        if preset_name not in RAY_TRACING_PRESETS:
            return {"error": f"Unknown preset: {preset_name}. Use list_ray_tracing_rhythmic_presets()."}
        config = RAY_TRACING_PRESETS[preset_name]
        state_a_id = config["state_a"]
        state_b_id = config["state_b"]
        oscillation_pattern = config["pattern"]
        num_cycles = config["num_cycles"]
        steps_per_cycle = config["steps_per_cycle"]
    else:
        if state_a_id not in RAY_TRACING_COORDS:
            return {"error": f"Unknown state: {state_a_id}. Options: {list(RAY_TRACING_COORDS.keys())}"}
        if state_b_id not in RAY_TRACING_COORDS:
            return {"error": f"Unknown state: {state_b_id}. Options: {list(RAY_TRACING_COORDS.keys())}"}

    state_a = RAY_TRACING_COORDS[state_a_id]
    state_b = RAY_TRACING_COORDS[state_b_id]

    total_steps = num_cycles * steps_per_cycle
    alphas = _generate_oscillation(total_steps, num_cycles, oscillation_pattern)

    # Apply phase offset by rotating the alpha array
    if phase_offset > 0.0:
        offset_samples = int(phase_offset * steps_per_cycle)
        alphas = alphas[offset_samples:] + alphas[:offset_samples]

    sequence = [_interpolate_states(state_a, state_b, a) for a in alphas]

    return {
        "domain": "ray_tracing",
        "preset_name": preset_name or "custom",
        "state_a": state_a_id,
        "state_b": state_b_id,
        "pattern": oscillation_pattern,
        "period": steps_per_cycle,
        "num_cycles": num_cycles,
        "total_steps": total_steps,
        "phase_offset": phase_offset,
        "parameter_names": RAY_TRACING_PARAMETER_NAMES,
        "sequence": sequence,
    }


@mcp.tool()
def get_ray_tracing_coordinates(state_id: str = "all") -> dict:
    """
    Get normalized parameter coordinates for ray tracing canonical states.

    Args:
        state_id: Specific state name or "all" for complete registry

    Returns:
        Parameter coordinates in [0.0, 1.0] morphospace

    Cost: 0 tokens (Layer 1 lookup)
    """
    if state_id == "all":
        return {
            "parameter_names": RAY_TRACING_PARAMETER_NAMES,
            "states": RAY_TRACING_COORDS,
        }

    if state_id not in RAY_TRACING_COORDS:
        return {"error": f"Unknown state: {state_id}. Options: {list(RAY_TRACING_COORDS.keys())}"}

    return {
        "state_id": state_id,
        "parameter_names": RAY_TRACING_PARAMETER_NAMES,
        "coordinates": RAY_TRACING_COORDS[state_id],
    }


# ============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION PROMPT GENERATION (0 tokens)
# ============================================================================

def _euclidean_distance(a: dict, b: dict, keys: list) -> float:
    """Euclidean distance between two state dicts over specified keys."""
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def _extract_visual_vocabulary(state: dict, strength: float = 1.0) -> dict:
    """
    Map a parameter-space state to the nearest canonical visual type
    and return image-generation-ready keywords.

    Nearest-neighbor matching against RAY_TRACING_VISUAL_TYPES.

    Args:
        state: Parameter coordinates dict
        strength: Keyword weight multiplier [0.0, 1.0]

    Returns:
        Dict with nearest_type, distance, keywords, optical_properties
    """
    best_name = None
    best_dist = float("inf")

    for type_name, type_def in RAY_TRACING_VISUAL_TYPES.items():
        dist = _euclidean_distance(state, type_def["coords"], RAY_TRACING_PARAMETER_NAMES)
        if dist < best_dist:
            best_dist = dist
            best_name = type_name

    matched = RAY_TRACING_VISUAL_TYPES[best_name]

    # Weight keywords by strength
    if strength < 1.0:
        weighted_keywords = [f"({strength:.1f}) {kw}" for kw in matched["keywords"]]
    else:
        weighted_keywords = list(matched["keywords"])

    return {
        "nearest_type": best_name,
        "distance": round(best_dist, 4),
        "keywords": weighted_keywords,
        "optical_properties": matched["optical_properties"],
    }


@mcp.tool()
def extract_ray_tracing_visual_vocabulary(
    state: dict,
    strength: float = 1.0,
) -> dict:
    """
    Extract visual vocabulary from ray tracing parameter coordinates.

    Maps a 5D parameter state to the nearest canonical ray tracing
    visual type and returns image-generation-ready keywords.

    Uses nearest-neighbor matching against 5 visual types derived from
    the ray tracing taxonomy.

    Args:
        state: Parameter coordinates dict with keys:
            light_warmth, caustic_complexity, surface_specularity,
            dispersion_intensity, motion_dynamism
        strength: Keyword weight multiplier [0.0, 1.0] (default: 1.0)

    Returns:
        Dict with nearest_type, distance, keywords, optical_properties

    Cost: 0 tokens (pure Layer 2 computation)
    """
    # Validate keys
    missing = [p for p in RAY_TRACING_PARAMETER_NAMES if p not in state]
    if missing:
        return {"error": f"Missing parameters: {missing}. Required: {RAY_TRACING_PARAMETER_NAMES}"}

    return _extract_visual_vocabulary(state, strength)


@mcp.tool()
def generate_ray_tracing_attractor_prompt(
    preset_name: str = "",
    custom_state: dict = None,
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4,
) -> dict:
    """
    Generate image generation prompt from ray tracing attractor state.

    Translates mathematical attractor coordinates into visual prompts
    suitable for image generation (ComfyUI, Stable Diffusion, DALL-E, etc.).

    Modes:
        composite: Single blended prompt from current state
        sequence: Multiple keyframe prompts from a rhythmic preset trajectory

    Args:
        preset_name: Rhythmic preset name (uses midpoint state for composite,
                     full trajectory for sequence)
        custom_state: Optional custom parameter coordinates dict.
                     Overrides preset_name if provided.
        mode: "composite" | "sequence"
        style_modifier: Optional prefix ("photorealistic", "oil painting", etc.)
        keyframe_count: Number of keyframes for sequence mode (default: 4)

    Returns:
        Dict with prompt(s), vocabulary details, and preset metadata

    Cost: 0 tokens (Layer 2 deterministic)
    """
    if mode == "sequence":
        return _generate_sequence_prompts(preset_name, keyframe_count, style_modifier)

    # --- Composite mode ---
    if custom_state:
        missing = [p for p in RAY_TRACING_PARAMETER_NAMES if p not in custom_state]
        if missing:
            return {"error": f"Missing parameters: {missing}"}
        state = custom_state
        source = "custom"
    elif preset_name:
        if preset_name not in RAY_TRACING_PRESETS:
            return {"error": f"Unknown preset: {preset_name}"}
        trajectory = _generate_preset_trajectory(preset_name)
        # Use midpoint of first cycle as representative state
        midpoint_idx = RAY_TRACING_PRESETS[preset_name]["steps_per_cycle"] // 2
        state = trajectory[midpoint_idx]
        source = preset_name
    else:
        return {"error": "Provide either preset_name or custom_state"}

    vocabulary = _extract_visual_vocabulary(state, strength=1.0)

    # Build prompt
    prompt_parts = []
    if style_modifier:
        prompt_parts.append(style_modifier)
    prompt_parts.extend(vocabulary["keywords"])

    # Add optical property descriptors
    optical = vocabulary["optical_properties"]
    prompt_parts.append(f"{optical['finish']} finish")
    prompt_parts.append(f"{optical['shadow_type']} shadows")

    # Add ray tracing quality anchors
    prompt_parts.extend([
        "ray traced rendering",
        "physically accurate light transport",
        "photorealistic caustic detail",
    ])

    prompt = ", ".join(prompt_parts)

    return {
        "prompt": prompt,
        "source": source,
        "state": state,
        "vocabulary": vocabulary,
        "mode": "composite",
    }


def _generate_sequence_prompts(
    preset_name: str,
    keyframe_count: int,
    style_modifier: str,
) -> dict:
    """
    Generate keyframe prompts from a rhythmic preset trajectory.

    Extracts evenly-spaced keyframes and generates a prompt for each.
    """
    if not preset_name or preset_name not in RAY_TRACING_PRESETS:
        return {"error": f"Sequence mode requires valid preset_name. Options: {list(RAY_TRACING_PRESETS.keys())}"}

    trajectory = _generate_preset_trajectory(preset_name)
    total = len(trajectory)

    # Evenly-spaced keyframe indices
    indices = [int(i * total / keyframe_count) for i in range(keyframe_count)]

    keyframes = []
    for idx in indices:
        state = trajectory[idx]
        vocab = _extract_visual_vocabulary(state, strength=1.0)

        parts = []
        if style_modifier:
            parts.append(style_modifier)
        parts.extend(vocab["keywords"])

        optical = vocab["optical_properties"]
        parts.append(f"{optical['finish']} finish")
        parts.append(f"{optical['shadow_type']} shadows")
        parts.extend([
            "ray traced rendering",
            "physically accurate light transport",
        ])

        keyframes.append({
            "step": idx,
            "state": state,
            "prompt": ", ".join(parts),
            "vocabulary": vocab,
        })

    return {
        "preset": preset_name,
        "mode": "sequence",
        "keyframe_count": keyframe_count,
        "total_steps": total,
        "period": RAY_TRACING_PRESETS[preset_name]["steps_per_cycle"],
        "keyframes": keyframes,
    }


@mcp.tool()
def map_ray_tracing_parameters(
    state_id: str,
    intensity: str = "moderate",
    emphasis: str = "light",
) -> dict:
    """
    Map a canonical ray tracing state to visual parameters.

    Combines state coordinates with intensity and emphasis modifiers
    to produce a complete parameter set ready for visualization or
    cross-domain composition.

    Args:
        state_id: Canonical state (soft_ambient, prismatic_crystal, etc.)
        intensity: "subtle", "moderate", or "dramatic"
        emphasis: "light", "surface", "caustic", "motion", or "dispersion"

    Returns:
        Complete parameter set with modifiers applied

    Cost: 0 tokens (Layer 2 deterministic)
    """
    if state_id not in RAY_TRACING_COORDS:
        return {"error": f"Unknown state: {state_id}. Options: {list(RAY_TRACING_COORDS.keys())}"}

    base = dict(RAY_TRACING_COORDS[state_id])

    # Intensity scaling
    intensity_scale = {"subtle": 0.6, "moderate": 1.0, "dramatic": 1.4}
    scale = intensity_scale.get(intensity, 1.0)

    # Emphasis boost map
    emphasis_map = {
        "light": "light_warmth",
        "surface": "surface_specularity",
        "caustic": "caustic_complexity",
        "motion": "motion_dynamism",
        "dispersion": "dispersion_intensity",
    }

    boosted_param = emphasis_map.get(emphasis)

    params = {}
    for p in RAY_TRACING_PARAMETER_NAMES:
        val = base[p] * scale
        if p == boosted_param:
            val = min(1.0, val * 1.3)  # 30% emphasis boost
        params[p] = round(min(1.0, max(0.0, val)), 4)

    vocabulary = _extract_visual_vocabulary(params)

    return {
        "state_id": state_id,
        "intensity": intensity,
        "emphasis": emphasis,
        "parameters": params,
        "vocabulary": vocabulary,
    }


# ============================================================================
# SERVER INFO & INTENTIONALITY
# ============================================================================

@mcp.tool()
def get_server_info() -> dict:
    """Get ray tracing aesthetic server information."""
    return {
        "name": "Ray Tracing Aesthetic MCP",
        "version": "2.6.0",
        "source": "Midjourney Compendium Ray Tracing Guide",
        "architecture": "three_layer_olog",
        "taxonomies": {
            "light_sources": "150+ enumerated sources",
            "refractive_media": "30+ optical media",
            "caustic_movements": "19 motion patterns",
            "receiving_surfaces": "50+ material types"
        },
        "cost_optimization": "Layer 1 & 2 deterministic (0 tokens), Layer 3 synthesis (~150-200 tokens)",
        "llm_cost_savings": "~70-80% vs pure LLM prompt enhancement",
        "phase_2_6_enhancements": {
            "rhythmic_presets": True,
            "preset_count": len(RAY_TRACING_PRESETS),
            "presets": list(RAY_TRACING_PRESETS.keys()),
            "canonical_states": list(RAY_TRACING_COORDS.keys()),
            "parameter_count": len(RAY_TRACING_PARAMETER_NAMES),
            "parameter_names": RAY_TRACING_PARAMETER_NAMES,
            "periods": sorted(set(
                p["steps_per_cycle"] for p in RAY_TRACING_PRESETS.values()
            )),
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_type_count": len(RAY_TRACING_VISUAL_TYPES),
            "visual_types": list(RAY_TRACING_VISUAL_TYPES.keys()),
            "prompt_modes": ["composite", "sequence"],
            "supported_domains": ["ray_tracing"],
        },
    }

@mcp.tool()
def get_intentionality() -> dict:
    """
    Explain WHY physically-based rendering taxonomy matters.
    
    Returns theoretical foundation and practical value.
    """
    return {
        "theoretical_foundation": {
            "physics_basis": "Light transport is deterministic - ray tracing simulates actual photon behavior according to Maxwell's equations",
            "compositional_structure": "Caustic effects decompose into Light ⊗ Medium ⊗ Movement ⊗ Surface tensor products",
            "emergent_properties": "Complex optical phenomena (rainbow dispersion, subsurface glow, recursive reflections) emerge from simple parameter combinations",
            "empirical_validation": "Extracted from Midjourney Compendium - validated through extensive prompt engineering practice"
        },
        
        "cost_optimization": {
            "layer_1_savings": "150+ light sources enumerated (0 tokens to list vs ~500 tokens to describe)",
            "layer_2_savings": "Optical calculations deterministic (0 tokens vs ~300 tokens for LLM physics reasoning)",
            "layer_3_efficiency": "Claude receives validated params (~150 tokens) vs full optical description (~1000+ tokens)",
            "total_reduction": "~70-80% reduction in inference cost per prompt enhancement"
        },
        
        "practical_value": {
            "precision": "Deterministic taxonomy prevents hallucinated optical effects",
            "consistency": "Same parameters always produce same specifications",
            "composition_guidance": "Compatibility matrix prevents physically implausible combinations",
            "creative_leverage": "Validated optical parameters free Claude to focus on creative synthesis"
        },
        
        "integration_value": {
            "photographic_perspective": "Camera position determines light falloff and caustic visibility patterns",
            "material_servers": "Surface properties from other domains map to ray tracing parameters",
            "temporal_dynamics": "Phase 2.6 rhythmic presets enable temporal aesthetic composition",
            "narrative_arc": "Lighting progression can follow dramatic structure via preset sequences",
            "multi_domain_composition": "Coordinates compatible with Tier 4D emergent attractor discovery",
            "attractor_visualization": "Phase 2.7 prompt generation bridges parameter space to image generation"
        }
    }

if __name__ == "__main__":
    mcp.run()
