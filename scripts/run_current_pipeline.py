"""Current delivery: VR-v1-b. Old recipes: run_legacy_pipeline.py."""
if __package__:
    from .run_vr_v1_b_pipeline import main
else:
    from run_vr_v1_b_pipeline import main

if __name__ == '__main__':
    main()
