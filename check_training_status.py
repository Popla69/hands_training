"""
Quick script to check current training status
"""
import json
import os
from datetime import datetime
import pandas as pd

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)

# Check if training state exists
if os.path.exists('models_tf2/training_state.json'):
    with open('models_tf2/training_state.json', 'r') as f:
        state = json.load(f)
    
    print(f"\n📊 Current Training Status:")
    print(f"  Last Completed Epoch: {state['last_epoch']}")
    print(f"  Best Validation Accuracy: {state['best_val_accuracy']:.2%}")
    print(f"  Last Update: {state['timestamp']}")
    print(f"  Checkpoint: {state['checkpoint_path']}")
    
    # Check if checkpoint exists
    if os.path.exists(state['checkpoint_path']):
        size_mb = os.path.getsize(state['checkpoint_path']) / (1024 * 1024)
        print(f"  Checkpoint Size: {size_mb:.1f} MB")
        print(f"  ✅ Checkpoint file exists")
    else:
        print(f"  ⚠️  Checkpoint file not found!")
else:
    print("\n⚠️  No training state found")
    print("Training may not have started yet")

# Check training log
if os.path.exists('models_tf2/training_log.csv'):
    print(f"\n📈 Training Progress:")
    df = pd.read_csv('models_tf2/training_log.csv')
    
    print(f"  Total Epochs Completed: {len(df)}")
    print(f"  Current Training Accuracy: {df['accuracy'].iloc[-1]:.2%}")
    print(f"  Current Validation Accuracy: {df['val_accuracy'].iloc[-1]:.2%}")
    print(f"  Current Training Loss: {df['loss'].iloc[-1]:.4f}")
    print(f"  Current Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    
    # Show trend
    if len(df) >= 5:
        recent_val_acc = df['val_accuracy'].tail(5).mean()
        print(f"\n  Recent 5-epoch avg val accuracy: {recent_val_acc:.2%}")
        
        # Check if improving
        if len(df) >= 10:
            prev_5 = df['val_accuracy'].iloc[-10:-5].mean()
            recent_5 = df['val_accuracy'].tail(5).mean()
            improvement = recent_5 - prev_5
            
            if improvement > 0:
                print(f"  📈 Improving! (+{improvement:.2%} vs previous 5 epochs)")
            elif improvement < -0.01:
                print(f"  📉 Declining ({improvement:.2%} vs previous 5 epochs)")
            else:
                print(f"  ➡️  Stable (±{abs(improvement):.2%})")
    
    # Show last 5 epochs
    print(f"\n  Last 5 Epochs:")
    print(df[['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss']].tail(5).to_string(index=False))
    
else:
    print("\n⚠️  No training log found")

# Check for running processes
print(f"\n🔍 Process Check:")
try:
    import psutil
    python_procs = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']) 
                    if 'python' in p.info['name'].lower()]
    
    if python_procs:
        print(f"  ✅ Found {len(python_procs)} Python process(es) running")
        for proc in python_procs:
            mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
            print(f"     PID {proc.info['pid']}: CPU {proc.info['cpu_percent']:.1f}%, Memory {mem_mb:.0f} MB")
    else:
        print(f"  ⚠️  No Python processes found - training may have stopped")
except ImportError:
    print(f"  ℹ️  Install psutil for process monitoring: pip install psutil")
except Exception as e:
    print(f"  ⚠️  Could not check processes: {e}")

# Recommendations
print(f"\n💡 Recommendations:")

if os.path.exists('models_tf2/training_state.json'):
    with open('models_tf2/training_state.json', 'r') as f:
        state = json.load(f)
    
    val_acc = state['best_val_accuracy']
    
    if val_acc < 0.70:
        print("  - Accuracy is low. Training needs more time.")
        print("  - Consider training for at least 50 epochs")
    elif val_acc < 0.85:
        print("  - Good progress! Keep training.")
        print("  - Target: 85%+ for production use")
    elif val_acc < 0.90:
        print("  - Great accuracy! Almost production-ready.")
        print("  - Consider training a bit more for 90%+")
    else:
        print("  - Excellent accuracy! Model is production-ready!")
        print("  - You can stop training or continue for marginal gains")
    
    # Check if training is stalled
    if os.path.exists('models_tf2/training_log.csv'):
        df = pd.read_csv('models_tf2/training_log.csv')
        if len(df) >= 10:
            recent_std = df['val_accuracy'].tail(10).std()
            if recent_std < 0.005:  # Very little variation
                print("  ⚠️  Validation accuracy has plateaued")
                print("  - Consider stopping training (early stopping)")
                print("  - Or reduce learning rate")

print("\n" + "="*70)
