# Ego–Human Two-Branch Factorized Dreamer

The current factorized path is selected with `model=factorized_dreamer`.  The
original Dreamer implementation remains available.  This version intentionally
does not consume an environment point cloud, BEV, PointPillars feature or Env
RSSM state.

## Inputs

- Ego14: episode-local position, body-heading velocity/acceleration, AGL
  altitude, roll/pitch and episode-relative yaw sin/cos.
- Human root `[B,T,N,10]`: body-frame root position/velocity, person extent and
  confidence.
- Root-relative joints `[B,T,N,J,7]`: joint position/velocity minus that
  person's root, plus confidence.  The axes still use the drone body frame.
- Goal position `[B,T,3]`: fixed target in the episode-start local frame.
- Goal feature `[B,T,8]`: current body-relative delta, distance, unit direction
  and normalized heading error.

## Real posterior step

1. Ego14 is encoded by an MLP.
2. Root MLP and causal per-person ST-GCN are fused into one token per Human.
3. Content-only sparse asymmetric attention is applied to observations:
   Ego reads Ego plus every valid Human; Human n reads only Ego and itself.
4. The contextual Ego/Human observation tokens update two separate RSSM
   posteriors.  Human slots share Human-RSSM parameters but keep separate state.
5. Before every prior transition, a separately parameterized sparse attention
   couples the current latent states with the same visibility rule.  Its
   context is concatenated with the drone action before each branch transition.
6. Policy attention receives `[Action, Goal, Ego, Human_1 ... Human_N]` and only
   the Action-token output is sent to Actor, Critic, Reward and Continue.

There is no private/relation split, explicit 10-D relation feature, relation
adapter or relation-only Q/K path.  All Q/K/V values come from normally encoded
token content and both attention blocks retain residual plus FFN residual paths.

## Imagination

Imagination never invokes observation encoders.  At every step it:

1. decodes current Ego14 from the imagined Ego latent;
2. recomputes the body-relative Goal feature from the fixed target;
3. runs Goal-conditioned policy attention and samples an action;
4. advances the conditionally coupled Ego/Human priors.

The Goal conditions policy, reward and continuation readout; it is not fed into
the physical RSSM transitions.

## Main tensor shapes

| Tensor | Shape |
|---|---|
| Ego observation token | `[B,T,1,D]` |
| Human observation tokens | `[B,T,N,D]` |
| Ego deter/stoch | `[B,T,Dd]`, `[B,T,S,K]` |
| Human deter/stoch | `[B,T,N,Dd]`, `[B,T,N,S,K]` |
| Observation attention input | `[B*T,1+N,D]` |
| Policy attention input | `[B,3+N,D]` |
| Actor input | `[B,D]`, exactly the Action-token output |

## Offline training

```bash
cd /path/to/omninxt-WM/world_model
python scripts/train_factorized_offline.py \
  --data-root /path/to/records \
  --pose-root /path/to/pose3d \
  --output /path/to/checkpoints/factorized.pt
```

No LiDAR/BEV cache argument is required.  The adapter reads Ego/Goal data from
frame JSON and strict 3-D pose caches for Humans.  Before production training,
fill `factorized.ego_state_mean/std` using training-split data only.

## Verification

```bash
cd /path/to/omninxt-WM/world_model
python -m unittest \
  tests.test_factorized_encoders \
  tests.test_factorized_rssm \
  tests.test_factorized_dreamer \
  tests.test_factorized_prediction_heads \
  tests.test_factorized_schema \
  tests.test_final_architecture_contract \
  tests.test_final_attentions -v
```
