#!/bin/bash
# Submit the (model x parser) grid. Each line is one job.
#
# Rules, stated in advance:
#   pymupdf  -> all four domains. Works uniformly; the model-comparison baseline.
#   docling  -> academic, credit, swimming. 10-K excluded: docling roughly
#               doubles those documents (e.g. csco 281k -> 400k chars) and both
#               models OOM on them at 600k max_input_chars.
#   camelot  -> swimming only. On prose it fabricates table structure and
#               splits words across cells (see parse_corpus output), so its
#               numbers on academic/credit would be parser artefacts.
set -e
cd ~/NLP_Lab

ALL="academic/research finance/credit_agreement finance/10kq sport/swimming"
NO10K="academic/research finance/credit_agreement sport/swimming"
SWIM="sport/swimming"

for M in qwen35_4b gemma3_4b; do
  sbatch -J "${M}_pym" run_grid_cell.sh $M pymupdf $ALL
  sbatch -J "${M}_doc" run_grid_cell.sh $M docling $NO10K
  sbatch -J "${M}_cam" run_grid_cell.sh $M camelot $SWIM
done
squeue --me