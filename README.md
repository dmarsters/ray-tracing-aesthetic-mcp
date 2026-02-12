# Ray Tracing Aesthetic MCP Server

Physically-based rendering parameter server extracted from the [Midjourney Compendium Ray Tracing Guide](https://www.midjourneycompendium.ch/subject/ray-tracing).

## Overview

This MCP server provides deterministic ray tracing aesthetic composition through a three-layer olog architecture:

- **Layer 1** (0 tokens): Pure taxonomy enumeration of 250+ optical components
- **Layer 2** (0 tokens): Deterministic composition validation and optical calculations
- **Layer 3** (~150-200 tokens): Synthesis context preparation for Claude

**Cost savings: ~70-80% reduction** in LLM inference costs vs pure prompt engineering.

## Core Taxonomies

### Light Sources (150+)
- Natural: celestial (sun/moon/stars), weather (lightning/fog), fire, bioluminescence
- Artificial: interior, exterior, neon, vehicle, stage, screen, specialty, industrial
- Each includes: color temperature (K), intensity, spectral distribution, shadow quality, caustic strength

### Refractive Media (30+)
- Water-based: underwater, puddles, waterfalls, mist
- Glass: greenhouse, stained glass, chandeliers, sculptures
- Ice: caves, frozen waterfalls, frost patterns
- Each includes: refractive index, dispersion, transparency, caustic shape

### Caustic Movements (19)
Dancing, flowing, rippling, cascading, pulsing, flickering, undulating, spiraling, fragmenting, kaleidoscoping, shimmering, morphing, radiating, converging, drifting, scattering, weaving, streaming, oscillating

### Receiving Surfaces (50+)
- Metallic: chrome, steel, car paint, foil
- Stone: marble, granite, asphalt
- Organic: skin, wood, snow, sand
- Fabric: satin, silk, velvet, leather
- Each includes: reflectivity, roughness, subsurface scattering, caustic visibility

## Compositional Algebra

Caustic effects decompose into tensor products:

```
CausticEffect = LightSource ⊗ Medium ⊗ Movement ⊗ Surface
```

The server validates compositions, detects incompatibilities, and identifies emergent effects like:
- Rainbow dispersion
- Subsurface scattering glow
- Recursive reflection amplification
- Volumetric caustic projection

## Installation

### Local Development
```bash
git clone <repository>
cd ray-tracing-aesthetic-mcp
pip install -e .
```

### FastMCP Cloud Deployment
```bash
# Deploy to FastMCP Cloud
fastmcp deploy ray_tracing_aesthetic_mcp/server.py:mcp
```

## Usage

### Basic Composition
```python
# List available light sources
sources = list_light_sources(category="natural_celestial")

# Compose caustic effect
result = compose_caustic_effect(
    light_source="noon_sunlight_direct",
    refractive_medium="underwater_scene",
    movement="rippling",
    receiving_surface="sand_white_beach"
)
```

### Synthesis Context for Image Generation
```python
# Prepare complete context for Claude
context = prepare_ray_tracing_synthesis_context(
    light_source="neon_sign_cyan_blue",
    refractive_medium="rain_puddle",
    movement="rippling",
    receiving_surface="asphalt_wet",
    subject="sports car",
    scene_description="urban night scene"
)

# Returns validated optical parameters, emergent effects,
# technical vocabulary, and synthesis guidelines
```

### Validation
```python
# Validate composition without movement
validation = validate_ray_tracing_composition(
    light_source="crystal_chandeliers",
    refractive_medium="glass_atrium",
    receiving_surface="marble_floor_polished"
)
```

## Tool Reference

### Layer 1 - Taxonomy Enumeration (0 tokens)
- `list_light_sources(category)` - 150+ sources with optical properties
- `list_refractive_media()` - 30+ media with refraction parameters
- `list_caustic_movements()` - 19 motion patterns
- `list_receiving_surfaces()` - 50+ surface types

### Layer 2 - Deterministic Composition (0 tokens)
- `compose_caustic_effect(light, medium, movement, surface)` - Full composition
- `validate_ray_tracing_composition(light, medium, surface)` - Compatibility check

### Layer 3 - Synthesis Context (~150-200 tokens)
- `prepare_ray_tracing_synthesis_context(...)` - Complete context for Claude

### Meta
- `get_server_info()` - Server capabilities and architecture
- `get_intentionality()` - Theoretical foundation and cost analysis

## Integration Bridges

This server is designed to compose with other Lushy MCP servers:

- **photographic-perspective-mcp**: Camera angle → light falloff patterns
- **material-property servers**: Surface descriptions → ray tracing parameters
- **temporal-dynamics**: Movement patterns → animation systems
- **narrative-arc**: Lighting progression → dramatic structure

## Examples

### Classic Underwater Caustics
```python
context = prepare_ray_tracing_synthesis_context(
    light_source="noon_sunlight_direct",
    refractive_medium="underwater_scene",
    movement="rippling",
    receiving_surface="sand_white_beach",
    subject="coral reef"
)
# → Natural organic caustic patterns on sandy bottom
```

### Luxury Automotive Showroom
```python
context = prepare_ray_tracing_synthesis_context(
    light_source="led_streetlights",  # or architectural_skylight
    refractive_medium="glass_atrium",
    movement="drifting",
    receiving_surface="car_bonnet_polished",
    subject="futuristic sports car"
)
# → Elegant geometric caustics on metallic surface
```

### Cyberpunk Urban Puddles
```python
context = prepare_ray_tracing_synthesis_context(
    light_source="neon_sign_multicolored",
    refractive_medium="rain_puddle",
    movement="rippling",
    receiving_surface="asphalt_wet",
    subject="city street at night"
)
# → Colored caustic reflections on wet pavement
```

## Architecture Notes

**Deterministic Operations (Layer 1 & 2):**
- No LLM calls for taxonomy listing or composition validation
- Pure Python calculations based on optical physics
- Compatibility checking via pre-defined rules matrix

**Claude Synthesis (Layer 3):**
- Receives validated optical parameters
- Gets technical vocabulary appropriate for image generators
- Focuses creative effort on aesthetic interpretation, not physics reasoning

**Cost Comparison:**
- Pure LLM prompt enhancement: ~1000-1500 tokens per composition
- This server: ~150-200 tokens (Claude synthesis only)
- Savings: ~70-80% reduction in inference cost

## Data Source

All taxonomies extracted from:
- [Midjourney Compendium - Ray Tracing Guide](https://www.midjourneycompendium.ch/subject/ray-tracing)
- Comprehensive catalog of physically-based rendering parameters
- Validated through extensive Midjourney prompt engineering practice

## License

Taxonomy data extracted from publicly available Midjourney Compendium.
Server implementation: MIT License

## Version

1.0.0 - Initial release with complete taxonomies from Midjourney Compendium
