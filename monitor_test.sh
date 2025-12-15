#!/bin/bash
# Monitor trading test progress

cd /Users/ajwashington/pulsegithub

echo "=== Trading RL Test Progress Monitor ==="
echo "Started monitoring at: $(date)"
echo ""

while true; do
    if pgrep -f "trading_rl_test" > /dev/null; then
        echo "[$(date +%H:%M:%S)] Test is running..."
        tail -5 /tmp/trading_test_output.txt 2>/dev/null | grep -E "Episode|Trial|Sharpe|completed|PulseOS" || echo "  (no recent output)"
        echo ""
        sleep 30
    else
        echo "[$(date +%H:%M:%S)] Test completed or stopped!"
        echo ""
        echo "Final output:"
        tail -50 /tmp/trading_test_output.txt
        break
    fi
done




