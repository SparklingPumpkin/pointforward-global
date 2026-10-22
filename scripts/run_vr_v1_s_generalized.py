"""Reuse the frozen VR-v1-s pipeline with the requested fresh stopping policy.

Only recipe/config location changes; source, assets and all other stages retain
the delivered implementation. Replay still loads the original accepted state.
"""
import run_vr_v1_s_pipeline as pipeline

pipeline.RELEASE = pipeline.PROJECT/'configs/runtime-stop'
main = pipeline.main

if __name__ == '__main__':
    main()
