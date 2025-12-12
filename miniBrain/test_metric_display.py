#!/usr/bin/env python3
"""
Test to verify that metrics display properly in the animation
"""
import numpy as np
import sys
import os
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lab import metrics, lz_complexity_fast

def test_metric_display():
    """Test that metrics display properly when updated every frame"""
    
    print("Testing metric display fix...")
    
    # Create test states like in the animation
    state_a = {
        'R_hist': [],
        'cached_entropy': 0.0,
        'cached_lyap': 0.0,
        'cached_lz': 0.0,
        'step_count': 0,
        'x': np.random.randn(100)
    }
    
    # Simulate some R values
    np.random.seed(42)
    for i in range(50):
        # Generate some realistic R values
        R_val = 0.5 + 0.3 * np.sin(i * 0.1) + 0.1 * np.random.randn()
        state_a['R_hist'].append(R_val)
        state_a['step_count'] = i
        
        # Test metric update logic (every frame now)
        if len(state_a['R_hist']) >= 10:
            recent_R = np.array(state_a['R_hist'][-30:])
            state_a['cached_entropy'] = metrics.cached_entropy(recent_R, "A")
            state_a['cached_lyap'] = metrics.cached_lyapunov(recent_R, "A")
            
            if state_a['step_count'] > 10:
                # Use dynamic binary sequence from recent R values instead of static neuron states
                if len(recent_R) >= 10:
                    binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                    state_a['cached_lz'] = lz_complexity_fast(binary_from_r)
                else:
                    binary_state = (state_a['x'] > 0).astype(int)[:200]
                    state_a['cached_lz'] = lz_complexity_fast(binary_state)
                state_a['cached_lz'] = lz_complexity_fast(binary_state)
        
        # Check if metrics are non-zero after sufficient data
        if i >= 15:  # Should have updated metrics by now
            print(f"Step {i}: Entropy={state_a['cached_entropy']:.3f}, "
                  f"Lyapunov={state_a['cached_lyap']:.3f}, "
                  f"LZ={state_a['cached_lz']:.2f}")
            
            # Verify metrics are being calculated
            assert state_a['cached_entropy'] > 0, f"Entropy should be > 0, got {state_a['cached_entropy']}"
            assert state_a['cached_lyap'] > 0, f"Lyapunov should be > 0, got {state_a['cached_lyap']}"
            assert state_a['cached_lz'] > 0, f"LZ should be > 0, got {state_a['cached_lz']}"
    
    print("OK All metrics are displaying properly!")
    print(f"Final metrics - Entropy: {state_a['cached_entropy']:.3f}, "
          f"Lyapunov: {state_a['cached_lyap']:.3f}, "
          f"LZ: {state_a['cached_lz']:.2f}")
    
    return True

if __name__ == "__main__":
    print("Testing metric display fix...\n")
    test_metric_display()
    print("\nOK TEST COMPLETE - Metrics display properly when updated every frame!")