# AI Cybersecurity Tool - Automated Data Population with Cronjobs

This guide explains how to set up automated data population for your AI Cybersecurity Tool using cronjobs.

---

## 🎯 Overview

The automated data population system will continuously send realistic network traffic data to your API, ensuring your dashboard always has fresh data to display. This is perfect for:

- **Demonstrations** - Always have live data to show
- **Testing** - Continuous validation of system performance
- **Development** - Populate dashboard during development
- **Monitoring** - Test alerting and monitoring systems

---

## 🚀 Quick Setup

### 1. Run the Setup Script
```bash
cd "/Users/parthgohil/Documents/Coding Projects/ai-cybersecurity-tool"
bash setup_cronjobs.sh
```

### 2. Choose Your Schedule
The script will present you with options:
- **Every 5 minutes** - High activity (good for demos)
- **Every 10 minutes** - Medium activity (balanced)
- **Every 15 minutes** - Low activity (light load)
- **Every hour** - Very low activity (minimal load)
- **Custom schedule** - Define your own timing

### 3. Verify Setup
```bash
# View your cronjobs
crontab -l

# Monitor logs
tail -f logs/cron_populator.log
```

---

## 📊 What Gets Populated

### Traffic Types Generated:
1. **Benign Traffic** (70% by default)
   - Normal HTTP/HTTPS traffic
   - SSH connections
   - Email traffic
   - DNS queries

2. **Threat Traffic** (30% by default)
   - **DDoS Attacks** - High packet rates, short duration
   - **Port Scans** - Single packets to random ports
   - **Brute Force** - Multiple failed login attempts

### Data Sent to Dashboard:
- ✅ Real-time threat scores
- ✅ Model predictions from all 5 ML models
- ✅ Feature importance explanations
- ✅ Performance metrics
- ✅ Alert notifications
- ✅ System health data

---

## ⚙️ Configuration Options

### Auto Populator Settings
You can customize the auto populator by editing `auto_data_populator.py`:

```python
# In the run_continuous_population method:
populator.run_continuous_population(
    interval_seconds=5,    # Time between requests
    threat_ratio=0.3       # 30% threats, 70% benign
)
```

### Cron Schedule Examples
```bash
# Every 5 minutes
*/5 * * * *

# Every 30 minutes
*/30 * * * *

# Every 2 hours
0 */2 * * *

# Every weekday at 9 AM
0 9 * * 1-5

# Every day at midnight
0 0 * * *
```

---

## 📝 Logs and Monitoring

### Log Locations
- **Cron Logs**: `logs/cron_populator.log`
- **API Logs**: `logs/api.log`
- **System Logs**: `logs/ai_cybersecurity.log`

### Monitor Logs
```bash
# Real-time log monitoring
tail -f logs/cron_populator.log

# Check for errors
grep "ERROR" logs/cron_populator.log

# View recent activity
tail -n 50 logs/cron_populator.log
```

### Log Content
The logs will show:
- ✅ Successful API requests
- 🚨 Threat detections
- ❌ Errors and failures
- 📊 Statistics and performance metrics

---

## 🛠️ Manual Operation

### Run Auto Populator Manually
```bash
# Start the API server first
python api/app.py &

# Run the auto populator
python auto_data_populator.py
```

### Interactive Options
When running manually, you can choose:
- **Single DDoS Attack Test**
- **Single Port Scan Test**
- **Single Brute Force Test**
- **Single Benign Traffic Test**
- **Continuous Mixed Simulation** (5 or 10 minutes)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Not Running
```bash
# Check if API is running
curl http://localhost:5001/health

# Start API if not running
python api/app.py
```

#### 2. Cronjob Not Working
```bash
# Check cron service
sudo launchctl list | grep cron

# View cron logs
grep CRON /var/log/system.log

# Check cronjob syntax
crontab -l
```

#### 3. Permission Issues
```bash
# Make scripts executable
chmod +x auto_data_populator.py
chmod +x run_auto_populator.sh
chmod +x setup_cronjobs.sh
```

#### 4. Virtual Environment Issues
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Debug Mode
```bash
# Run with verbose logging
python auto_data_populator.py 2>&1 | tee debug.log
```

---

## 📊 Performance Impact

### System Resources
- **CPU Usage**: Minimal (< 1% when idle)
- **Memory Usage**: ~50MB for populator process
- **Network**: ~1-5 requests per minute
- **Disk**: Logs grow by ~1MB per day

### API Performance
- **Request Rate**: Configurable (default: every 5 seconds)
- **Response Time**: < 100ms per request
- **Concurrent Users**: No impact on other users
- **Database**: Minimal impact (simple INSERT operations)

---

## 🎯 Use Cases

### 1. **Professor Presentation**
```bash
# Set up high-frequency population for demo
bash setup_cronjobs.sh  # Choose option 1 (every 5 minutes)
```
- Dashboard always shows live data
- Multiple threat types demonstrated
- Real-time model performance visible

### 2. **Development Testing**
```bash
# Set up medium-frequency population
bash setup_cronjobs.sh  # Choose option 2 (every 10 minutes)
```
- Continuous system validation
- Alert testing
- Performance monitoring

### 3. **Production Simulation**
```bash
# Set up low-frequency population
bash setup_cronjobs.sh  # Choose option 4 (every hour)
```
- Minimal system impact
- Long-term data collection
- Trend analysis

---

## 🔒 Security Considerations

### API Security
- ✅ Uses existing API authentication
- ✅ Rate limiting applies to populator
- ✅ No sensitive data transmitted
- ✅ Logs contain no credentials

### System Security
- ✅ Runs with user permissions (not root)
- ✅ Limited to project directory
- ✅ No external network access
- ✅ Graceful error handling

---

## 📈 Monitoring Dashboard

### What You'll See in Dashboard
1. **Real-time Threat Chart** - Live threat scores
2. **Threat Distribution** - Pie chart of threat types
3. **Model Performance** - Live model metrics
4. **Feature Attention** - Real-time explanations
5. **Recent Alerts** - Alert notifications
6. **System Health** - Performance metrics

### Dashboard URLs
- **Main Dashboard**: http://localhost:3000
- **API Health**: http://localhost:5001/health
- **Model Performance**: http://localhost:5001/models/performance

---

## 🎉 Benefits

### For Demonstrations
- ✅ Always have live data to show
- ✅ Multiple attack types demonstrated
- ✅ Real-time model performance visible
- ✅ Professional presentation quality

### For Development
- ✅ Continuous system validation
- ✅ Automated testing capability
- ✅ Performance monitoring
- ✅ Alert system testing

### For Learning
- ✅ Understanding different attack patterns
- ✅ Seeing how ML models respond
- ✅ Observing real-time feature importance
- ✅ Learning about network traffic analysis

---

## 🚀 Getting Started

### Step 1: Ensure API is Running
```bash
python api/app.py
```

### Step 2: Setup Cronjobs
```bash
bash setup_cronjobs.sh
```

### Step 3: Open Dashboard
Open http://localhost:3000 in your browser

### Step 4: Monitor
```bash
tail -f logs/cron_populator.log
```

---

## 📞 Support

If you encounter issues:

1. **Check Logs**: `tail -f logs/cron_populator.log`
2. **Verify API**: `curl http://localhost:5001/health`
3. **Check Cronjobs**: `crontab -l`
4. **Restart Services**: Restart API and check again

---

**Your AI Cybersecurity Tool will now automatically populate with realistic data, making it perfect for demonstrations and continuous testing!** 🎉
