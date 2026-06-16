"""GLSL shader sources for the OpenXR viewer."""

_WORLD_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# World-space overlay fragment shader (plain RGBA texture, no parallax)
# u_alpha scales the output alpha; defaults to 1.0 (fully opaque per texture).
_OVERLAY_FRAG = """
#version 330
uniform sampler2D tex;
uniform float u_alpha;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, uv);
    fragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

# Fullscreen swizzle blit: copies an RGBA texture into a target that the
# compositor reads as BGRA. Used by the EXT_memory_object interop path when the
# OpenXR runtime hands us a BGRA swapchain.
_BLIT_FRAG = """
#version 330
uniform sampler2D u_src;
uniform int u_swap_rb;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(u_src, uv);
    fragColor = (u_swap_rb != 0) ? c.bgra : c;
}
"""

# Solid-color vertex shader (no UV avoids GLSL optimizer stripping in_uv)
_SOLID_VERT = """
#version 330
in vec2 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# Solid-color fragment shader for the screen border quad
_SOLID_FRAG = """
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""

# 3D vertex shader for tapered rainbow beam
_BEAM_VERT = """
#version 330
in vec3 in_position;
in float in_v;
out float v_v;
uniform mat4 u_mvp;
void main() {
    v_v = in_v;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_BEAM_FRAG = """
#version 330
in float v_v;
out vec4 fragColor;
uniform float u_time;
void main() {
    // Rainbow gradient: blue→cyan→green→yellow→red, flowing from root to tip
    float t = fract(v_v + u_time * 0.4);
    vec3 col;
    if (t < 0.167)      col = mix(vec3(0.0,0.4,1.0), vec3(0.0,1.0,1.0), t/0.167);
    else if (t < 0.333) col = mix(vec3(0.0,1.0,1.0), vec3(0.0,1.0,0.0), (t-0.167)/0.166);
    else if (t < 0.5)   col = mix(vec3(0.0,1.0,0.0), vec3(1.0,1.0,0.0), (t-0.333)/0.167);
    else if (t < 0.667) col = mix(vec3(1.0,1.0,0.0), vec3(1.0,0.5,0.0), (t-0.5)/0.167);
    else if (t < 0.833) col = mix(vec3(1.0,0.5,0.0), vec3(1.0,0.0,0.0), (t-0.667)/0.166);
    else                col = mix(vec3(1.0,0.0,0.0), vec3(0.0,0.4,1.0), (t-0.833)/0.167);
    fragColor = vec4(col, 1.0);
}
"""

_CURVED_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

# VR controller model shader (improved: supports Blinn-Phong lighting and texture toggle)
_CTRL_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;   // Corresponds to 12 skipped bytes in the data
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_normal;
out vec3 v_position;
uniform mat4 u_mvp;
uniform mat4 u_model; // Used for normal transformation
void main() {
    v_uv = in_uv;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal; // Normal transformation
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    gl_Position = u_mvp * world_pos;
}
"""

_CTRL_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform vec3 u_light_color;    // Light source color
uniform vec3 u_ambient_color;  // Ambient light color
uniform vec3 u_base_color_factor; // Base color factor
uniform int u_use_texture;     // 0: use solid color, 1: sample texture
uniform vec3 u_camera_pos;     // Camera world coordinates (= headset position)

void main() {
    // Discard back faces (inner walls), keep only front faces (outer shell)
    if (!gl_FrontFacing) discard;

    vec3 N = normalize(v_normal);
    vec3 light_pos = u_camera_pos + vec3(0.0, 0.05, 0.0);
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(u_camera_pos - v_position);
    vec3 H = normalize(L + V);

    vec3 baseColor;
    if (u_use_texture == 1) {
        baseColor = texture(u_tex, v_uv).rgb * u_base_color_factor;
    } else {
        baseColor = u_base_color_factor;
    }

    float diff = abs(dot(N, L));
    vec3 diffuse = u_light_color * diff;
    vec3 ambient = u_ambient_color;
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = u_light_color * spec;

    fragColor = vec4((ambient + diffuse + specular) * baseColor, 1.0);
}
"""

_ENV_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec4 in_tangent;  // xyz + bitangent_sign (glTF §3.7.4)
out vec2 v_uv;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_tangent;
out float v_bitangent_sign;
uniform mat4 u_mvp;
uniform mat4 u_model;
void main() {
    v_uv = in_uv;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_tangent = normalize(mat3(transpose(inverse(u_model))) * in_tangent.xyz);
    v_bitangent_sign = in_tangent.w;
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    gl_Position = u_mvp * world_pos;
}
"""

_ENV_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;
in vec3 v_tangent;
in float v_bitangent_sign;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform sampler2D u_normal_tex;    // normal map (texture unit 4)
uniform sampler2D u_occlusion_tex; // occlusion map (texture unit 5)
uniform sampler2D u_mr_tex;        // metallicRoughness (texture unit 6: B=metal, G=rough)
uniform sampler2D u_emissive_tex;  // emissive map (texture unit 7)
uniform vec3 u_light_color;
uniform vec3 u_ambient_color;
uniform vec3 u_base_color_factor;
uniform int u_use_texture;
uniform int u_use_normal_tex;
uniform float u_normal_scale;
uniform int u_use_occlusion_tex;
uniform float u_occlusion_strength;
uniform vec3 u_camera_pos;
uniform float u_roughness;
uniform float u_metallic;
uniform vec3 u_emissive_factor;
uniform int u_unlit;             // KHR_materials_unlit: skip lighting
uniform float u_alpha_cutoff;    // alphaMode=MASK discard threshold
uniform int u_alpha_mode;        // 0=OPAQUE, 1=MASK, 2=BLEND
uniform int u_use_mr_tex;        // 0: use uniform factors, 1: sample mr texture
uniform int u_use_emissive_tex;  // 0: use factor only, 1: sample emissive texture
uniform vec2 u_tex_offset;       // KHR_texture_transform offset
uniform vec2 u_tex_scale;        // KHR_texture_transform scale
uniform vec3 u_light_dir;        // KHR_lights_punctual directional light
uniform vec3 u_light_intensity;  // light_color * intensity for directional light

// Fill lights (viewer-side point lights with range attenuation)
uniform int   u_fill_light_enabled0; // 0=skip, 1=evaluate
uniform vec3  u_fill_light_pos0;
uniform vec3  u_fill_light_color0;
uniform float u_fill_light_range0;
uniform int   u_fill_light_enabled1;
uniform vec3  u_fill_light_pos1;
uniform vec3  u_fill_light_color1;
uniform float u_fill_light_range1;

// Post-processing controls
uniform float u_env_exposure;
uniform float u_env_gamma;
uniform float u_emissive_strength;
uniform float u_base_alpha;

// --- Cinema "bias light" rectangular area light ----------------------------
// The virtual screen acts as an emissive rectangular source (analogous to a
// real TV's ambient bias light, e.g. Philips Hue Play behind a display).  We
// follow the Meta Horizon lighting guidelines:
//   * Emissive material + Area light (cf. Lighting Types table)
//   * Lambertian diffuse response so transitions are GRADUAL (a key Meta
//     guideline for VR comfort no harsh brightness flicker)
//   * Forward 180° hemisphere only (light cannot wrap behind the screen)
//   * Single fragment-shader light, no extra texture samples preserves the
//     ≥90 FPS target Meta sets for VR.
uniform int   u_screen_light_enabled;       // 0 = skip, 1 = evaluate
uniform vec3  u_screen_light_pos;           // world-space rectangle centre
uniform vec3  u_screen_light_normal;        // world-space forward (toward viewer)
uniform vec2  u_screen_light_half_size;     // half-width / half-height (metres)
uniform vec3  u_screen_light_color;         // sampled emissive colour (frame avg)
uniform float u_screen_light_intensity;     // master multiplier (e.g. 1.5)

const float PI = 3.14159265359;

// Fresnel-Schlick
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX / Trowbridge-Reitz normal distribution
float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith GGX geometry (Schlick)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// Reusable PBR evaluation for a single light source.
// Extracts the inline Cook-Torrance GGX math so fill lights can share it
// without duplicating the entire BRDF per light.
vec3 pbrLight(vec3 N, vec3 V, vec3 baseColor, float metallic, float roughness,
              vec3 L, vec3 lightColor, float attenuation) {
    if (attenuation <= 0.0 || length(lightColor) <= 0.001) return vec3(0.0);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3 H = normalize(L + V);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(VdotH, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;
    return (diffuse + specular) * lightColor * NdotL * attenuation;
}

// Soft range attenuation: 1/(1 + 4*(d/r)^2) — cheap (2 muls, 1 add, 1 rcp).
float softRangeAttenuation(float dist, float rangeMeters) {
    float r = max(rangeMeters, 0.001);
    float x = dist / r;
    return 1.0 / (1.0 + x * x * 4.0);
}

void main() {
    // KHR_texture_transform: compute transformed UV (glTF spec)
    vec2 t_uv = v_uv * u_tex_scale + u_tex_offset;

    // Compute texture alpha early for MASK discard and BLEND output.
    float texAlpha = (u_use_texture == 1) ? texture(u_tex, t_uv).a : 1.0;
    float materialAlpha = clamp(texAlpha * u_base_alpha, 0.0, 1.0);

    // alphaMode MASK: early discard (glTF spec 3.9.4)
    if (u_alpha_mode == 1 && u_use_texture == 1) {
        if (materialAlpha < u_alpha_cutoff) discard;
    }

    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;

    // Normal map perturbation (glTF MikkTSpace tangent)
    if (u_use_normal_tex == 1) {
        vec3 nm = texture(u_normal_tex, t_uv).rgb * 2.0 - 1.0;
        nm.xy *= u_normal_scale;
        nm = normalize(nm);
        // Use TANGENT attribute if available, else Gram-Schmidt fallback
        vec3 T = length(v_tangent) > 0.001 ? normalize(v_tangent) : normalize(cross(vec3(0.0, 1.0, 0.0), N));
        vec3 B = normalize(cross(N, T)) * v_bitangent_sign;
        N = normalize(T * nm.x + B * nm.y + N * nm.z);
    }

    vec3 baseColor;
    if (u_use_texture == 1) {
        baseColor = texture(u_tex, t_uv).rgb * u_base_color_factor;
    } else {
        baseColor = u_base_color_factor;
    }

    float metallic = clamp(u_metallic, 0.0, 1.0);
    float roughness = clamp(u_roughness, 0.04, 1.0);
    // metallicRoughnessTexture: B=metallic, G=roughness (glTF spec §3.9.4)
    if (u_use_mr_tex == 1) {
        vec3 mr = texture(u_mr_tex, t_uv).rgb;
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }

    vec3 light_pos = u_camera_pos + vec3(0.0, 0.05, 0.0);
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(u_camera_pos - v_position);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // PBR: dielectric F0 = 0.04, metals use baseColor as F0
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    // Cook-Torrance specular BRDF
    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(VdotH, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Diffuse: dielectrics scatter, metals have zero diffuse
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;

    // KHR_materials_unlit: skip all lighting, output baseColor directly
    if (u_unlit == 1) {
        float alpha = (u_alpha_mode == 2 || u_base_alpha < 0.999) ? materialAlpha : 1.0;
        fragColor = vec4(baseColor, alpha);
        return;
    }

    // Head-lamp point light
    vec3 Lo = (diffuse + specular) * u_light_color * NdotL;

    // Directional light (KHR_lights_punctual)
    if (length(u_light_intensity) > 0.001) {
        float NdotL_d = max(dot(N, -u_light_dir), 0.0);
        vec3 H_d = normalize(-u_light_dir + V);
        float D_d = DistributionGGX(max(dot(N, H_d), 0.0), roughness);
        float G_d = GeometrySmith(NdotV, NdotL_d, roughness);
        vec3 F_d = fresnelSchlick(max(dot(V, H_d), 0.0), F0);
        vec3 s_d = (D_d * G_d * F_d) / max(4.0 * NdotV * NdotL_d, 0.001);
        vec3 kD_d = (vec3(1.0) - F_d) * (1.0 - metallic);
        vec3 d_d = kD_d * baseColor / PI;
        Lo += (d_d + s_d) * u_light_intensity * NdotL_d;
    }

    // Fill light 0 (viewer-side point light with range attenuation)
    if (u_fill_light_enabled0 == 1) {
        vec3 toFill0 = u_fill_light_pos0 - v_position;
        float fillDist0 = length(toFill0);
        if (fillDist0 > 0.001) {
            Lo += pbrLight(N, V, baseColor, metallic, roughness,
                           toFill0 / fillDist0,
                           u_fill_light_color0,
                           softRangeAttenuation(fillDist0, u_fill_light_range0));
        }
    }

    // Fill light 1
    if (u_fill_light_enabled1 == 1) {
        vec3 toFill1 = u_fill_light_pos1 - v_position;
        float fillDist1 = length(toFill1);
        if (fillDist1 > 0.001) {
            Lo += pbrLight(N, V, baseColor, metallic, roughness,
                           toFill1 / fillDist1,
                           u_fill_light_color1,
                           softRangeAttenuation(fillDist1, u_fill_light_range1));
        }
    }

    // --- Cinema bias light (rectangular area light = the virtual screen) ---
    // Closed-form-ish approximation: use the rectangle centre as a single
    // representative point (Frostbite-style "most representative point" is
    // overkill here env surfaces are far enough that a point approximation
    // already matches a true rect to within a few %, while being ~10× cheaper
    // than the integral).  Meta's guideline is "use real-time lights sparingly"
    // and "minimize calculations" so we do exactly one area sample.
    if (u_screen_light_enabled == 1) {
        vec3  S_to_P  = v_position - u_screen_light_pos;       // from screen to frag
        float d       = length(S_to_P);
        vec3  L_s     = S_to_P / max(d, 0.001);                // direction screen to frag
        // 1) FORWARD-HEMISPHERE TEST (screen front side only)
        //    A real screen emits light only out of its front face, so we
        //    require the fragment to lie in the screen's forward half-space
        //    (dot(normal, screen->frag) > 0).  Use a smoothstep instead of
        //    a hard step so the grazing-angle transition is gradual --
        //    per Meta's lighting guideline, "avoid sudden changes in
        //    brightness".  Zero lower bound: no light behind the screen
        //    plane.  The 0.3 upper bound keeps the transition gradual
        //    for surfaces nearly in-plane with the screen.
        float front   = smoothstep(0.0, 0.3, dot(u_screen_light_normal, L_s));
        if (front > 0.0) {
            // 2) LAMBERTIAN cosine on the receiving surface
            float NdotL_s = max(dot(N, -L_s), 0.0);
            if (NdotL_s > 0.0) {
                // 3) DISTANCE FALL-OFF with smooth windowing.  Pure 1/d is
                //    physically correct but produces harsh hotspots near the
                //    screen undesirable per Meta's "avoid harsh transitions"
                //    rule.  We use the standard "windowed inverse square"
                //    f(d) = (1 / (d + r))   where r is the falloff knee.
                //    A LARGER r pushes the knee outward, so the light keeps
                //    its strength further into the room before rolling off
                //    (attn = 0.5 at d = r, 0.2 at d ≥2r, 0.1 at d ≥3r).
                //    Tuned to ~2× the half-diagonal so a 2.4 m screen still
                //    contributes meaningfully out to ~5 m, matching the
                //    user-perceived "bias light" spread of a real OLED.
                float half_diag = length(u_screen_light_half_size);
                float r0        = max(half_diag * 2.0, 0.50);
                float attn      = (r0 * r0) / (d * d + r0 * r0);
                // 4) AREA scale.  A rectangle of area A subtends an apparent
                //    solid angle that grows with A approximate this as
                //    A / (PI * d) for the diffuse term (small-angle limit) and
                //    clamp the near-field to avoid singularities (Meta:
                //    smooth/comfortable response).  The near-field clamp uses
                //    half_diag (not r0) so the close-up brightness is not
                //    suppressed by the wider falloff knee chosen above.
                float area      = 4.0 * u_screen_light_half_size.x
                                       * u_screen_light_half_size.y;
                float r_near    = max(half_diag * 0.5, 0.10);
                float area_term = area / (PI * max(d * d, r_near * r_near));
                // Suppress the close-in ring around the screen itself.
                // Keeps spill on room objects, but avoids a visible halo
                // hugging screen edge and nearby wall surfaces.
                float halo_free = smoothstep(
                    max(half_diag * 0.35, 0.75),
                    max(half_diag * 0.95, 1.75),
                    d
                );
                // 5) Final radiance contribution.  Diffuse-only on the receiver
                //    (kD * baseColor / PI is the Lambertian BRDF already
                //    computed above; we reuse it for spectral neutrality with
                //    the head-lamp path), then modulated by the screen colour
                //    and a user-tunable intensity gain.
                vec3  E_s = u_screen_light_color
                              * u_screen_light_intensity
                              * front * NdotL_s * attn * area_term * halo_free;
                Lo += diffuse * E_s * PI;   // PI cancels the 1/PI in `diffuse`
            }
        }
    }

    // Ambient
    float ao = 1.0;
    if (u_use_occlusion_tex == 1) {
        ao = mix(1.0, texture(u_occlusion_tex, t_uv).r, u_occlusion_strength);
    }
    vec3 ambient = u_ambient_color * baseColor * ao;

    vec3 emissive = u_emissive_factor;
    if (u_use_emissive_tex == 1) {
        emissive *= texture(u_emissive_tex, t_uv).rgb;
    }
    emissive *= u_emissive_strength;

    vec3 color = (Lo + ambient + emissive) * u_env_exposure;
    // HDR->LDR: Reinhard-like soft tonemap
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0 / max(u_env_gamma, 0.001)));

    float alpha = (u_alpha_mode == 2 || u_base_alpha < 0.999) ? materialAlpha : 1.0;
    fragColor = vec4(color, alpha);
}
"""

# Glow fragment shader: renders a soft glow outside a centered rectangle
_GLOW_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec2 u_screen_half;   // screen half-size in UV space
uniform vec3 u_glow_color;
uniform float u_glow_inv_range;
uniform float u_glow_inv_density_range;
uniform float u_glow_intensity;
void main() {
    vec2 d = abs(uv - 0.5) - u_screen_half;
    // Exterior distance to the screen rectangle (0 inside).
    vec2 edge = max(d, vec2(0.0));
    float hi = max(edge.x, edge.y);
    float lo = min(edge.x, edge.y);
    float dist = hi + lo * 0.375;  // cheap length approximation for soft glow
    if (dist <= 0.001) {
        discard;
    }
    // Finite cubic falloff avoids the old full-quad exp() + dither cost while
    // keeping the visible bias-light shoulder close to the previous glow.
    float x = clamp(1.0 - dist * u_glow_inv_range, 0.0, 1.0);
    float glow = x * x * x * u_glow_intensity;
    // Pixel density decays from 100% at the screen edge to 0% at
    // 0.75 * glow_width. This creates a sparse sunshine-like tail and avoids
    // blending lots of nearly invisible pixels.
    float density = clamp(1.0 - dist * u_glow_inv_density_range, 0.0, 1.0);
    vec2 p = fract(gl_FragCoord.xy * vec2(0.1031, 0.1030));
    p += vec2(dot(p, p.yx + 33.33));
    float grain = fract((p.x + p.y) * p.x);
    if (grain > density) {
        discard;
    }
    if (glow <= 0.001) {
        discard;
    }
    glow = min(glow, 1.0);
    frag_color = vec4(u_glow_color * glow, glow);
}
"""
