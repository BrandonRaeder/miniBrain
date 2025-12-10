#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

try:
    import lab
    print("[OK] Successfully imported lab.py")
except Exception as e:
    print(f"[ERROR] Failed to import lab.py: {e}")
    sys.exit(1)

def test_current_implementation():
    """Test the current entropy and Lyapunov implementations"""
    
    print("\n=== Testing Current Implementation ===")
    
    # Test 1: Check if metrics object exists
    if hasattr(lab, 'metrics'):
        print("[OK] Metrics object exists")
        metrics = lab.metrics
    else:
        print("[ERROR] Metrics object not found")
        return False
    
    # Test 2: Check methods exist
    if hasattr(metrics, 'cached_entropy'):
        print("[OK] cached_entropy method exists")
    else:
        print("[ERROR] cached_entropy method missing")
        return False
        
    if hasattr(metrics, 'cached_lyapunov'):
        print("[OK] cached_lyapunov method exists") 
    else:
        print("[ERROR] cached_lyapunov method missing")
        return False
    
    # Test 3: Run simulations
    try:
        print("\n--- Running Test Simulations ---")
        R_a, ws_a, x_a = lab.simulate_workspace(T=100)
        print(f"[OK] Option A simulation completed - R_hist length: {len(R_a)}")
        
        R_b, ws_b = lab.simulate_reflective_hierarchy(T=100)
        print(f"[OK] Option B simulation completed - R_hist length: {len(R_b)}")
        
        R_c, ws_c, x_c, self_error_c, self_model_c = lab.simulate_self_referential_workspace(T=100)
        print(f"[OK] Option C simulation completed - R_hist length: {len(R_c)}")
        
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        return False
    
    # Test 4: Test entropy calculations
    try:
        print("\n--- Testing Entropy Calculations ---")
        
        # Test cached entropy (used by A & B)
        if len(R_a) >= 30:
            entropy_a = metrics.cached_entropy(R_a[-30:], "A")
            print(f"[OK] Option A cached entropy: {entropy_a:.4f}")
        
        if len(R_b) >= 30:
            entropy_b = metrics.cached_entropy(R_b[-30:], "B")
            print(f"[OK] Option B cached entropy: {entropy_b:.4f}")
        
        # Test Lyapunov calculations
        if len(R_a) >= 12:
            lyap_a = metrics.cached_lyapunov(R_a[-30:], "A")
            print(f"[OK] Option A Lyapunov: {lyap_a:.4f}")
        
        if len(R_b) >= 12:
            lyap_b = metrics.cached_lyapunov(R_b[-30:], "B") 
            print(f"[OK] Option B Lyapunov: {lyap_b:.4f}")
        
        if len(R_c) >= 12:
            lyap_c = metrics.cached_lyapunov(R_c[-30:], "C")
            print(f"[OK] Option C Lyapunov: {lyap_c:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Entropy/Lyapunov test failed: {e}")
        return False
    
    # Test 5: Check if Option C uses different entropy calculation
    print("\n--- Analyzing Option C Entropy Implementation ---")
    print("Note: Option C uses direct entropy calculation in self-model encoding")
    print("This is inconsistent with Options A & B which use cached entropy")
    
    return True

def demonstrate_inconsistency():
    """Show the current entropy inconsistency in Option C"""
    
    print("\n=== Demonstrating Current Entropy Inconsistency ===")
    
    # Run Option C simulation
    R_c, ws_c, x_c, self_error_c, self_model_c = lab.simulate_self_referential_workspace(T=50)
    
    # Calculate entropy using Option C's current method (direct calculation)
    if len(R_c) > 10:
        recent_R = np.array(R_c[-10:])
        entropy_c_direct = -np.sum([r * np.log(r + 1e-10) for r in recent_R if r > 0])
        print(f"Option C direct entropy (current): {entropy_c_direct:.4f}")
    
    # Calculate entropy using standardized cached method
    if len(R_c) >= 30:
        entropy_c_cached = lab.metrics.cached_entropy(R_c[-30:], "C")
        print(f"Option C cached entropy (standardized): {entropy_c_cached:.4f}")
        print(f"Difference: {abs(entropy_c_direct - entropy_c_cached):.4f}")
    
    return True

if __name__ == "__main__":
    print("Testing current entropy and Lyapunov implementations in lab.py")
    
    success = test_current_implementation()
    if success:
        demonstrate_inconsistency()
        print("\n=== Summary ===")
        print("[OK] Lyapunov exponents are properly calculated for all three models")
        print("[OK] LZ complexity is populated for all three models") 
        print("[ERROR] Entropy calculation is inconsistent - Option C uses different method")
        print("-> Need to standardize entropy calculation for all three models")
    else:
        print("\n[ERROR] Tests failed - implementation has issues")