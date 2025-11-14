# 📋 How to View Logs - Quick Reference

## 🎯 **View Logs When Clicking "Send Test Data"**

### **Real-Time Log Viewing (Recommended)**

Open a **new terminal window** and run:

```bash
# Navigate to project directory
cd "/Users/parthgohil/Documents/Coding Projects/ai-cybersecurity-tool"

# View API logs in real-time (follows new entries)
tail -f logs/api.log
```

**What you'll see:**
- API requests coming in
- Model predictions
- Threat detections
- Any errors or warnings

---

### **Alternative: View Specific Log Files**

```bash
# Main API log (all activity)
tail -f logs/api.log

# API errors only
tail -f logs/api_errors.log

# API startup log (model loading info)
tail -f logs/api_startup.log

# All cybersecurity logs
tail -f logs/ai_cybersecurity.log
```

---

### **View Recent Log Entries**

```bash
# Last 50 lines of API log
tail -50 logs/api.log

# Last 100 lines with timestamps
tail -100 logs/api.log | grep -E "2025-11-12|prediction|threat|error"

# Search for specific terms
grep -i "test data\|send test\|prediction" logs/api.log | tail -20
```

---

### **Filter Logs for Specific Events**

```bash
# Only show predictions
tail -f logs/api.log | grep -i "prediction\|threat"

# Only show errors
tail -f logs/api.log | grep -i "error\|failed\|warning"

# Only show new model activity
tail -f logs/api.log | grep -i "new\|iot\|cicapt"

# Show Isolation Forest issues
tail -f logs/api.log | grep -i "isolation"
```

---

### **View Logs in Browser (if using log viewer)**

Some systems have web-based log viewers. Check if your setup includes one.

---

## 📊 **What to Look For When Testing**

When you click "Send Test Data", you should see:

1. **API Request Received:**
   ```
   INFO - API request received: POST /predict
   ```

2. **Model Predictions:**
   ```
   INFO - Model rf prediction: 0.85
   INFO - Model xgboost prediction: 0.75
   ```

3. **Threat Detection:**
   ```
   INFO - THREAT_DETECTED: {"threat_level": "Critical", "threat_score": 0.80, ...}
   ```

4. **Database Storage:**
   ```
   INFO - Stored threat in database
   ```

5. **Any Errors:**
   ```
   WARNING - Model isolation_forest prediction failed: ...
   ```

---

## 🔍 **Quick Commands Reference**

```bash
# Real-time API log
tail -f logs/api.log

# Last 20 lines
tail -20 logs/api.log

# Search for errors
grep -i error logs/api.log | tail -20

# Search for new models
grep -i "new\|iot" logs/api_startup.log

# Count total predictions
grep -c "prediction" logs/api.log

# View all log files
ls -lh logs/*.log
```

---

## 💡 **Pro Tip**

Keep two terminal windows open:
1. **Terminal 1:** Run `tail -f logs/api.log` (for real-time monitoring)
2. **Terminal 2:** Your regular terminal for other commands

When you click "Send Test Data" on the dashboard, watch Terminal 1 to see the logs appear in real-time!

---

## 🐛 **If Logs Don't Appear**

1. **Check if API is running:**
   ```bash
   curl http://localhost:5001/health
   ```

2. **Check log file permissions:**
   ```bash
   ls -l logs/api.log
   ```

3. **Check if logs directory exists:**
   ```bash
   ls -la logs/
   ```

4. **Restart API to regenerate logs:**
   ```bash
   # Stop API (Ctrl+C)
   # Then restart:
   python api/app.py
   ```

