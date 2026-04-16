#!/bin/bash
# Runs UCB, Thompson Sampling, and Epsilon-Greedy against China (gfwatch)
# then plots all three together with baselines for comparison.
#
# Usage:  bash run_comparison.sh
# Output: models/outputs/comparison_<timestamp>/plots/comparison_china.pdf

set -e

RLFOLDER="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH=$RLFOLDER

EPISODES=3
MEASUREMENTS=1000
GROUND_TRUTH=$RLFOLDER/inputs/gfwatch/gfwatch-blocklist.csv
TRANCO=$RLFOLDER/inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv
FEATURES="categories"
timestamp=$(date +%Y%m%d_%H%M%S)
OUTDIR=$RLFOLDER/models/outputs/comparison_${timestamp}
mkdir -p $OUTDIR

echo "=============================="
echo " CenRL Comparison Run"
echo " Measurements : $MEASUREMENTS"
echo " Episodes     : $EPISODES"
echo " Country      : China (gfwatch)"
echo " Output       : $OUTDIR"
echo "=============================="

cd $RLFOLDER/models

# UCB — best known params from paper
echo "[1/3] UCB (c=0.03)..."
python3 $RLFOLDER/models/ucb/ucb_naive.py \
  -m $MEASUREMENTS -E $EPISODES \
  -c=0.03 -s 0.0 -V 0.0 \
  -o $OUTDIR/ucb_naive_c0.03_stepsize0.0_initval0.0 \
  -g $GROUND_TRUTH -a $TRANCO -f $FEATURES 2>&1 | grep -E "Done with|episode"

# Thompson Sampling
echo "[2/3] Thompson Sampling..."
python3 $RLFOLDER/models/thompson_sampling/thompson_sampling.py \
  -m $MEASUREMENTS -E $EPISODES \
  -o $OUTDIR/thompson_sampling_c0_stepsize0.0_initval0.0 \
  -g $GROUND_TRUTH -a $TRANCO -f $FEATURES 2>&1 | grep -E "Done with|episode"

# Epsilon-Greedy (epsilon=0.1)
echo "[3/3] Epsilon-Greedy (e=0.1)..."
python3 $RLFOLDER/models/epsilon_greedy/epsilon_greedy_sampling.py \
  -m $MEASUREMENTS -E $EPISODES \
  -e=0.1 -s 0.0 -V 0.0 \
  -o $OUTDIR/epsilon_greedy_c0_stepsize0.0_initval0.0 \
  -g $GROUND_TRUTH -a $TRANCO -f $FEATURES 2>&1 | grep -E "Done with|episode"

echo "[4/5] Uncertainty Sampling (active learning)..."
python3 $RLFOLDER/models/active_learning/uncertainty_sampling.py \
  -m $MEASUREMENTS -E $EPISODES \
  -o $OUTDIR/uncertainty_sampling_c0_stepsize0.0_initval0.0 \
  -g $GROUND_TRUTH -a $TRANCO -f $FEATURES 2>&1 | grep -E "Done with|episode"

echo "[5/5] LinUCB (contextual bandit, alpha=0.5)..."
python3 $RLFOLDER/models/contextual_bandit/linucb.py \
  -m $MEASUREMENTS -E $EPISODES \
  -alpha 0.5 \
  -o $OUTDIR/linucb_c0_stepsize0.0_initval0.0 \
  -g $GROUND_TRUTH -a $TRANCO -f $FEATURES 2>&1 | grep -E "Done with|episode"

echo ""
echo "All models done. Plotting with baselines..."

cd $RLFOLDER/scripts
python3 plotter_dyn.py \
  --output_file_name comparison_china.pdf \
  --results_directory $OUTDIR \
  --results_prefix "" \
  --ground_truth_file_path $GROUND_TRUTH \
  --measurements $MEASUREMENTS \
  --episodes $EPISODES \
  --action_space_file_path $TRANCO \
  --include_baselines 2>&1 | tail -5

echo ""
echo "=============================="
echo " Done."
echo " Results : $OUTDIR"
echo " Plot    : $OUTDIR/plots/comparison_china.pdf"
echo "=============================="
