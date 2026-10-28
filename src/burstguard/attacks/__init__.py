"""Simulated compromised participants used to evaluate defenses (robustness study).

All attack variants follow published methods and operate only on the local training data / model update of a
simulated client inside the experiment:
  poison           data poisoning with a burst trigger and a scaled-down target (BackTime-style, Lin et al. 2024)
  scale            poisoning + update boosting by attack.boost (default n/m = model replacement, Bagdasaryan et
                   al. 2020; a smaller boost, e.g. 3, trades backdoor strength for clean accuracy = covert attacker)
  constrain_scale  poisoning + boosting projected to a norm bound (Sun et al. 2019; Bagdasaryan et al. 2020)
  neurotoxin       boosted poisoning with gradients restricted to rarely-updated coordinates (Zhang et al. 2022)
  adaptive         defense-aware boosted poisoning: only the exact trigger flips the forecast, random bursts inside
                   the screening search space keep clean labels (decoys), so a screen looking for generic bursts
                   has less signal; norm-bounded only with attack.adaptive_constrain = true
"""
