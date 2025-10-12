#!/bin/bash
# Setup Cronjobs for AI Cybersecurity Tool Auto Data Population

echo "🚀 AI CYBERSECURITY TOOL - CRONJOB SETUP"
echo "========================================"

# Get the current directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Project Directory: $PROJECT_DIR"

# Make the auto populator executable
chmod +x "$PROJECT_DIR/auto_data_populator.py"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Create a wrapper script for cron
cat > "$PROJECT_DIR/run_auto_populator.sh" << 'EOF'
#!/bin/bash
# Wrapper script for auto data population
# This script ensures proper environment setup for cron

# Set the project directory
PROJECT_DIR="/Users/parthgohil/Documents/Coding Projects/ai-cybersecurity-tool"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PATH="$PROJECT_DIR/venv/bin:$PATH"

# Run the auto populator
python auto_data_populator.py >> logs/cron_populator.log 2>&1
EOF

chmod +x "$PROJECT_DIR/run_auto_populator.sh"

echo ""
echo "📋 Available Cronjob Options:"
echo "1. Every 5 minutes (high activity)"
echo "2. Every 10 minutes (medium activity)"
echo "3. Every 15 minutes (low activity)"
echo "4. Every hour (very low activity)"
echo "5. Custom schedule"
echo "6. Remove all cronjobs"
echo ""

read -p "Choose an option (1-6): " choice

case $choice in
    1)
        CRON_SCHEDULE="*/5 * * * *"
        DESCRIPTION="every 5 minutes"
        ;;
    2)
        CRON_SCHEDULE="*/10 * * * *"
        DESCRIPTION="every 10 minutes"
        ;;
    3)
        CRON_SCHEDULE="*/15 * * * *"
        DESCRIPTION="every 15 minutes"
        ;;
    4)
        CRON_SCHEDULE="0 * * * *"
        DESCRIPTION="every hour"
        ;;
    5)
        echo ""
        echo "Custom cron schedule examples:"
        echo "  Every minute: * * * * *"
        echo "  Every 30 minutes: */30 * * * *"
        echo "  Every 2 hours: 0 */2 * * *"
        echo "  Every day at 9 AM: 0 9 * * *"
        echo "  Every weekday at 6 PM: 0 18 * * 1-5"
        echo ""
        read -p "Enter cron schedule: " CRON_SCHEDULE
        DESCRIPTION="custom schedule: $CRON_SCHEDULE"
        ;;
    6)
        echo "🗑️ Removing all AI Cybersecurity Tool cronjobs..."
        crontab -l 2>/dev/null | grep -v "run_auto_populator.sh" | crontab -
        echo "✅ All cronjobs removed"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Remove any existing cronjobs for this project
echo "🗑️ Removing existing cronjobs..."
crontab -l 2>/dev/null | grep -v "run_auto_populator.sh" | crontab -

# Add new cronjob
echo "➕ Adding new cronjob..."
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $PROJECT_DIR/run_auto_populator.sh") | crontab -

echo ""
echo "✅ Cronjob setup complete!"
echo "📅 Schedule: $DESCRIPTION"
echo "📁 Wrapper script: $PROJECT_DIR/run_auto_populator.sh"
echo "📝 Logs will be written to: $PROJECT_DIR/logs/cron_populator.log"
echo ""
echo "🔍 To view current cronjobs: crontab -l"
echo "📊 To monitor logs: tail -f $PROJECT_DIR/logs/cron_populator.log"
echo "🛑 To remove cronjobs: bash $PROJECT_DIR/setup_cronjobs.sh (choose option 6)"
echo ""
echo "⚠️  Important Notes:"
echo "   - Make sure your API server is running for the populator to work"
echo "   - The populator will run for 5 minutes each time it's triggered"
echo "   - Check logs if you don't see data in your dashboard"
echo ""
echo "🚀 Your AI Cybersecurity Tool will now automatically populate data $DESCRIPTION!"
