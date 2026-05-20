import os
import sys
import argparse

# Project path tracking
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

# Import individual validation run routines
from core.validators.likeness_validator import run_likeness_validation
from core.validators.blending_validator import run_blending_validation
from core.validators.spatial_integrity_validator import run_spatial_validation
from core.validators.specularity_validator import run_specularity_validation
from core.validators.realism_validator import run_realism_validation

def run_all_validation(output_dir, input_dir, reference_dir, models_dir=None):
    print("\n" + "="*100)
    print(" STARTING MASTER QUANTITATIVE PIPELINE CERTIFICATION HARNESS")
    print("="*100)
    print(f"Outputs Folder:    {output_dir}")
    print(f"Targets Folder:    {input_dir}")
    print(f"References Folder: {reference_dir}")
    
    # Run each suite and gather details
    likeness_res = run_likeness_validation(output_dir, input_dir, reference_dir, models_dir=models_dir)
    blending_res = run_blending_validation(output_dir, models_dir=models_dir)
    spatial_res = run_spatial_validation(output_dir, input_dir, models_dir=models_dir)
    specularity_res = run_specularity_validation(output_dir, models_dir=models_dir)
    realism_res = run_realism_validation(output_dir, models_dir=models_dir)
    
    print("\n" + "="*100)
    print(" PIPELINE Sprint v2.1 HARDENED - QUALITY CERTIFICATION REPORT")
    print("="*100)
    
    all_passed = True
    
    # Aggregate and present report card
    for out_name in likeness_res.keys():
        print(f"\n[REPORT CARD] File: {out_name}")
        print("-" * 60)
        
        # 1. Likeness Similarity
        avg_sim = likeness_res[out_name]["avg_sim"]
        max_sim = likeness_res[out_name]["max_sim"]
        struct_dev = likeness_res[out_name]["struct_dev"]
        likeness_verdict = likeness_res[out_name]["verdict"]
        print(f" - Likeness Avg / Max: {avg_sim:.4f} / {max_sim:.4f} (Target: >0.40 / >0.50) -> {likeness_verdict}")
        
        # 2. Spatial Integrity
        pct_altered = spatial_res.get(out_name, {}).get("pct_altered", 100.0)
        spatial_verdict = spatial_res.get(out_name, {}).get("verdict", "FAIL")
        print(f" - Spatial Integrity (BG Altered): {pct_altered:.4f}% (Target: <0.05%)       -> {spatial_verdict}")
        
        # 3. Blending Seamlessness
        blend_ratio = blending_res.get(out_name, {}).get("blending_ratio", 99.9)
        blending_verdict = blending_res.get(out_name, {}).get("verdict", "FAIL")
        print(f" - Blending Seam Ratio: {blend_ratio:.4f} (Target: <2.20)                   -> {blending_verdict}")
        
        # 4. Specularity & Lighting
        spec_ratio = specularity_res.get(out_name, {}).get("spec_ratio", 99.9)
        tone_ratio = specularity_res.get(out_name, {}).get("tone_ratio", 0.0)
        spec_verdict = specularity_res.get(out_name, {}).get("verdict", "FAIL")
        print(f" - Specularity Ratio (Face/Neck L): {spec_ratio:.4f} (Target: <1.15)        -> {spec_verdict}")
        print(f" - Skin Tone Balance Ratio: {tone_ratio:.4f}")
        
        # 5. Realism Texture Ratio
        texture_ratio = realism_res.get(out_name, {}).get("ratio", 0.0)
        print(f" - Sharpness Ratio (Face/BG): {texture_ratio:.4f}")
        
        # Final Verdict for this file
        file_passed = (
            likeness_verdict == "PASS" and 
            spatial_verdict == "PASS" and 
            blending_verdict == "PASS" and 
            spec_verdict == "PASS"
        )
        
        if file_passed:
            print(f" --> VERDICT STATUS: CERTIFIED PASS (Sprint v2.1 HARDENED)")
        else:
            print(f" --> VERDICT STATUS: REJECTED (Quality constraints failed)")
            all_passed = False
            
    print("\n" + "="*100)
    if all_passed:
        print(" MASTER STATUS: PIPELINE CERTIFIED -- SPRINT v2.1 HARDENED STABLE")
    else:
        print(" MASTER STATUS: CERTIFICATION REJECTED -- REGRESSION DETECTED")
    print("="*100 + "\n")
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Validator Suite")
    parser.add_argument("--output", default="samples/odiyan_swaps", help="Output directory containing results")
    parser.add_argument("--input", default="target_pics", help="Input directory containing targets")
    parser.add_argument("--references", default="odiyan_refs", help="Reference directory containing source faces")
    args = parser.parse_args()
    
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(proj_root, args.output)
    in_dir = args.input if os.path.isabs(args.input) else os.path.join(proj_root, args.input)
    ref_dir = args.references if os.path.isabs(args.references) else os.path.join(proj_root, args.references)
    
    run_all_validation(out_dir, in_dir, ref_dir)

if __name__ == '__main__':
    main()
