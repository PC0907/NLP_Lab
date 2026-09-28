# Documentation

## Shared

- [HPC guide](hpc.md): cluster setup, job submission and monitoring.
- [Troubleshooting](troubleshooting.md): known failure modes and fixes.

## Reasoning-trace track (DeepSeek-R1 on SOB)

Start with the [project overview](reasoning/PROJECT_OVERVIEW.md). For every module and design decision, see the
[deep dive](reasoning/DEEP_DIVE.md); for the pipeline and a map of result files, see the
[track guide](reasoning/REASONING_TRACK.md).

| Document | Topic |
|---|---|
| [Paper_Skeleton](reasoning/Paper_Skeleton.md) | Paper outline and figure plan |
| [Project_Master_Guide](reasoning/Project_Master_Guide.md) | Walkthrough of the whole project with examples |
| [Meeting_Briefing_2026-08-25](reasoning/Meeting_Briefing_2026-08-25.md) | Supervisor meeting briefing |
| [Update_04](reasoning/Update_04.md) → [Update_08](reasoning/Update_08.md) | Dated progress updates |

## ExtractBench track (Qwen3.5 and others on PDF extraction)

| Document | Topic |
|---|---|
| [update_01](extractbench/updates/update_01.md) → [update_10](extractbench/updates/update_10.md) | Dated progress updates (01–03 predate the track split and are shared) |
| [selective-regeneration](extractbench/selective-regeneration.md) | Selective regeneration: design, implementation, findings |
| [regen](extractbench/regen.md) | Verified selective-regeneration result (ExtractBench, 4B) |
| [docling](extractbench/docling.md) | Docling vs. PyMuPDF parsing |
| [dataset-issues](extractbench/dataset-issues.md) | RealKIE dataset evaluation: findings and path forward |
| [llama-issues](extractbench/llama-issues.md) | Llama 3.1 8B structural mismatch and label contamination |

Dated updates are kept as written at the time. They may use old file paths, old stage numbers, or numbers that were
later corrected; the most recent document on a topic supersedes earlier ones.
