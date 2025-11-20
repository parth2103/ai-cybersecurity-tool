import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Container, Grid, Paper, Typography, Card, CardContent,
  Alert, Box, LinearProgress, Chip, Button
} from '@mui/material';
import {
  Security, Speed, Shield, TrendingUp, Storage, CloudQueue,
  Warning, CheckCircle
} from '@mui/icons-material';
import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';
import AttentionVisualizer from './AttentionVisualizer';
import ModelPerformanceMonitor from './ModelPerformanceMonitor';
import LiveDataManager from './LiveDataManager';
import { chartColors } from '../theme';

const API_URL = 'http://localhost:5001';
const API_KEY = 'dev-key-123';

// Styled Metric Card Component with Enhanced Animations
const MetricCard = ({ title, value, icon: Icon, color = 'primary', subtitle }) => (
  <Card
    sx={{
      height: 110,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden',
      background: `linear-gradient(135deg, rgba(${color === 'error' ? '244, 67, 54' : color === 'warning' ? '255, 152, 0' : color === 'success' ? '76, 175, 80' : '33, 150, 243'}, 0.05) 0%, rgba(21, 26, 53, 0.9) 100%)`,
      border: `1px solid rgba(${color === 'error' ? '244, 67, 54' : color === 'warning' ? '255, 152, 0' : color === 'success' ? '76, 175, 80' : '33, 150, 243'}, 0.2)`,
      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '3px',
        background: `linear-gradient(90deg, ${chartColors[color]} 0%, transparent 100%)`,
        animation: 'shimmer 3s ease-in-out infinite',
      },
      '&::after': {
        content: '""',
        position: 'absolute',
        top: '50%',
        right: -20,
        width: 100,
        height: 100,
        borderRadius: '50%',
        background: `radial-gradient(circle, rgba(${color === 'error' ? '244, 67, 54' : color === 'warning' ? '255, 152, 0' : color === 'success' ? '76, 175, 80' : '33, 150, 243'}, 0.1) 0%, transparent 70%)`,
        transform: 'translateY(-50%)',
        animation: 'pulse 4s ease-in-out infinite',
      },
      '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: `0 8px 25px rgba(${color === 'error' ? '244, 67, 54' : color === 'warning' ? '255, 152, 0' : color === 'success' ? '76, 175, 80' : '33, 150, 243'}, 0.3)`,
        borderColor: `${chartColors[color]}60`,
      },
    }}
  >
    <CardContent sx={{ p: 2, position: 'relative', zIndex: 1 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 'bold', color: chartColors[color], mb: 0.5 }}>
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
              {subtitle}
            </Typography>
          )}
        </Box>
        <Icon sx={{ fontSize: 32, color: chartColors[color], opacity: 0.8 }} />
      </Box>
    </CardContent>
  </Card>
);

const LiveDashboard = () => {
  const [stats, setStats] = useState({
    current_threat_level: 'None',
    threat_history: [],
    threats_detected: 0,
    total_requests: 0
  });
  
  const [systemInfo, setSystemInfo] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    disk_usage: 0,
    models_loaded: [],
    total_predictions: 0,
    threats_detected: 0,
    detection_rate: 0
  });

  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [explainFeatures, setExplainFeatures] = useState(null);

  // Memoized callback functions to prevent infinite loops
  const handleStatsUpdate = useCallback((newStats) => {
    console.log('📊 Live stats update:', newStats);
    setStats(newStats);
  }, []);

  const handleAlertsUpdate = useCallback((newAlert) => {
    console.log('🚨 Live alert update:', newAlert);
    setAlerts(prev => {
      const prevArray = Array.isArray(prev) ? prev : [];
      return [newAlert, ...prevArray.slice(0, 9)]; // Keep last 10 alerts
    });
  }, []);

  const handleSystemUpdate = useCallback((newSystemInfo) => {
    console.log('💻 Live system update:', newSystemInfo);
    setSystemInfo(newSystemInfo);
  }, []);

  // Handle features ready for explanation
  const handleFeaturesReady = useCallback((features) => {
    console.log('📊 Features ready for explanation:', features);
    setExplainFeatures(features);
  }, []);

  // Live data manager for real-time updates
  const { isConnected, connectionStatus, sendTestData, isTestSending, toggleContinuousSending, isContinuousSending } = LiveDataManager({
    onStatsUpdate: handleStatsUpdate,
    onAlertsUpdate: handleAlertsUpdate,
    onSystemUpdate: handleSystemUpdate,
    onFeaturesReady: handleFeaturesReady
  });

  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`, {
        headers: { 'X-API-Key': API_KEY }
      });
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setError('Failed to fetch threat statistics');
    }
  }, []);

  const fetchSystemInfo = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/system/info`, {
        headers: { 'X-API-Key': 'admin-key-789' }
      });
      setSystemInfo(response.data);
    } catch (error) {
      console.error('Error fetching system info:', error);
    }
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/alerts`, {
        headers: { 'X-API-Key': API_KEY }
      });
      // Handle the response structure: { "alerts": [...] }
      const alertsData = response.data?.alerts || response.data || [];
      const alertsArray = Array.isArray(alertsData) ? alertsData : [];
      setAlerts(alertsArray);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      setAlerts([]); // Set empty array on error
    }
  }, []);

  useEffect(() => {
    // Initial load
    (async () => {
      try {
        setLoading(true);
        await Promise.all([fetchStats(), fetchSystemInfo(), fetchAlerts()]);
        setLoading(false);
      } catch (error) {
        console.error('Initial load error:', error);
        setLoading(false);
      }
    })();
  }, [fetchStats, fetchSystemInfo, fetchAlerts]);

  const getThreatLevelColor = (level) => {
    const colors = {
      'Critical': chartColors.error,
      'High': chartColors.warning,
      'Medium': chartColors.warning,
      'Low': chartColors.primary,
      'None': chartColors.success
    };
    return colors[level] || chartColors.primary;
  };

  const getThreatIcon = (level) => {
    switch (level) {
      case 'Critical': return <Security sx={{ color: chartColors.error }} />;
      case 'High': return <Warning sx={{ color: chartColors.warning }} />;
      case 'Medium': return <Warning sx={{ color: chartColors.warning }} />;
      case 'Low': return <Shield sx={{ color: chartColors.primary }} />;
      default: return <CheckCircle sx={{ color: chartColors.success }} />;
    }
  };

  // Prepare real-time chart data
  const realtimeData = stats.threat_history.slice(-20).map((entry, index) => ({
    time: new Date(entry.timestamp).toLocaleTimeString(),
    threat_score: entry.threat_score * 100,
    level: entry.threat_level
  }));

  // Prepare pie chart data
  const pieData = [
    { name: 'Safe', value: stats.threat_history.filter(t => t.threat_level === 'None').length, color: chartColors.success },
    { name: 'Threats', value: stats.threat_history.filter(t => t.threat_level !== 'None').length, color: chartColors.error }
  ];

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <LinearProgress sx={{ width: '100%', maxWidth: 400 }} />
          <Typography variant="h6" sx={{ ml: 2 }}>
            Loading live data...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 1.5, maxHeight: '100vh', overflow: 'auto' }}>
      {/* Header */}
      <Box mb={1.5}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ 
          fontWeight: 'bold', 
          background: 'linear-gradient(45deg, #2196F3, #21CBF3)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textAlign: 'center',
          mb: 1,
          fontSize: '1.75rem'
        }}>
          🛡️ AI Cybersecurity Dashboard
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
      </Box>

      <Grid container spacing={1.5}>
        {/* Threat Level Alert Banner - Compact */}
        <Grid item xs={12}>
          <Alert
            severity={stats.current_threat_level === 'Critical' ? 'error' :
                    stats.current_threat_level === 'High' ? 'warning' :
                    stats.current_threat_level === 'Medium' ? 'warning' : 'success'}
            icon={getThreatIcon(stats.current_threat_level)}
            sx={{
              borderRadius: 1.5,
              borderLeft: `3px solid ${getThreatLevelColor(stats.current_threat_level)}`,
              py: 0.25,
              background: stats.current_threat_level === 'Critical' || stats.current_threat_level === 'High'
                ? `linear-gradient(90deg, rgba(244, 67, 54, 0.15) 0%, transparent 100%)`
                : `linear-gradient(90deg, rgba(76, 175, 80, 0.1) 0%, transparent 100%)`,
              animation: stats.current_threat_level === 'Critical'
                ? 'pulse 2s ease-in-out infinite'
                : 'none',
              '@keyframes pulse': {
                '0%, 100%': {
                  boxShadow: `0 0 0 0 ${getThreatLevelColor(stats.current_threat_level)}40`,
                  borderLeftColor: getThreatLevelColor(stats.current_threat_level),
                },
                '50%': {
                  boxShadow: `0 0 20px 5px ${getThreatLevelColor(stats.current_threat_level)}20`,
                  borderLeftColor: getThreatLevelColor(stats.current_threat_level),
                },
              },
            }}
          >
            <Typography variant="body1" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
              {stats.current_threat_level === 'Critical' && '🚨 CRITICAL THREAT DETECTED!'}
              {stats.current_threat_level === 'High' && '⚠️ HIGH THREAT DETECTED'}
              {stats.current_threat_level === 'Medium' && '⚡ MEDIUM THREAT DETECTED'}
              {stats.current_threat_level === 'Low' && '🔍 LOW THREAT DETECTED'}
              {stats.current_threat_level === 'None' && '✅ SYSTEM SECURE'}
            </Typography>
          </Alert>
        </Grid>

        {/* Live Connection Status */}
        <Grid item xs={12}>
          <Alert 
            severity={isConnected ? 'success' : 'warning'}
            sx={{ 
              mb: 1.5,
              py: 0.5,
              background: isConnected 
                ? 'linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(21, 26, 53, 0.9) 100%)'
                : 'linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(21, 26, 53, 0.9) 100%)',
              border: isConnected 
                ? '1px solid rgba(76, 175, 80, 0.3)'
                : '1px solid rgba(255, 152, 0, 0.3)',
              borderRadius: 1.5
            }}
          >
            <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
              {isConnected ? <CheckCircle /> : <Warning />}
              <Typography variant="body1" sx={{ fontSize: '0.95rem' }}>
                Live Data: {connectionStatus}
              </Typography>
              <Chip 
                label={isConnected ? "Real-time WebSocket" : "Polling Mode"} 
                size="small" 
                color={isConnected ? "success" : "warning"}
                sx={{ ml: 1 }}
              />
              <Button
                variant={isContinuousSending ? "contained" : "outlined"}
                size="small"
                onClick={toggleContinuousSending}
                disabled={isTestSending}
                color={isContinuousSending ? "error" : "primary"}
                sx={{ 
                  ml: 2,
                  minWidth: '140px', // Fixed width to prevent layout shift
                  animation: isContinuousSending ? 'pulse 2s ease-in-out infinite' : 'none',
                  '@keyframes pulse': {
                    '0%, 100%': {
                      boxShadow: '0 0 0 0 rgba(244, 67, 54, 0.7)',
                    },
                    '50%': {
                      boxShadow: '0 0 0 10px rgba(244, 67, 54, 0)',
                    },
                  },
                }}
              >
                {isTestSending ? "Sending..." : isContinuousSending ? "⏹️ Stop Live Demo" : "▶️ Start Live Demo"}
              </Button>
            </Box>
          </Alert>
        </Grid>

        {/* Metric Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Requests"
            value={stats.total_requests}
            icon={Storage}
            color="primary"
            subtitle="API calls processed"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Threats Detected"
            value={stats.threats_detected}
            icon={Security}
            color="error"
            subtitle="Malicious activities"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Detection Rate"
            value={`${systemInfo.detection_rate.toFixed(1)}%`}
            icon={TrendingUp}
            color="success"
            subtitle="Accuracy"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="System Health"
            value={`${(100 - (systemInfo.cpu_percent || 0)).toFixed(0)}%`}
            icon={CloudQueue}
            color="success"
            subtitle="CPU availability"
          />
        </Grid>

        {/* Main Charts Section */}
        <Grid item xs={12} md={8}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: 320,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'rgba(33, 150, 243, 0.5)',
                boxShadow: '0 12px 40px rgba(33, 150, 243, 0.25), inset 0 0 40px rgba(33, 150, 243, 0.08)',
              },
            }}
          >
            <Typography
              variant="subtitle1"
              gutterBottom
              sx={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '0.95rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(33, 150, 243, 0.5)',
                mb: 1
              }}
            >
              <Speed sx={{ mr: 1, color: 'primary.main', fontSize: 18 }} />
              Real-time Threat Scores
            </Typography>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={realtimeData}>
                <defs>
                  <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartColors.error} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={chartColors.error} stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="rgba(255,255,255,0.7)"
                  fontSize={12}
                  tickLine={false}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)"
                  fontSize={12}
                  tickLine={false}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(21, 26, 53, 0.95)',
                    border: '1px solid rgba(33, 150, 243, 0.3)',
                    borderRadius: 8,
                    color: '#fff'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="threat_score"
                  stroke={chartColors.error}
                  fillOpacity={1}
                  fill="url(#colorThreat)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: 320,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
            }}
          >
            <Typography
              variant="subtitle1"
              gutterBottom
              sx={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '0.95rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(33, 150, 243, 0.5)',
                mb: 1
              }}
            >
              <Shield sx={{ mr: 1, color: 'primary.main', fontSize: 18 }} />
              Threat Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(21, 26, 53, 0.95)',
                    border: '1px solid rgba(33, 150, 243, 0.3)',
                    borderRadius: 8,
                    color: '#fff'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Performance Monitor and Feature Attention - Side by Side */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: 340,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
              overflow: 'auto'
            }}
          >
            <ModelPerformanceMonitor 
              apiBaseUrl={API_URL} 
              apiKey={API_KEY} 
            />
          </Paper>
        </Grid>

        {/* Feature Attention Visualizer */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: 340,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
              overflow: 'auto'
            }}
          >
            <AttentionVisualizer 
              apiBaseUrl={API_URL} 
              apiKey={API_KEY}
              features={explainFeatures}
            />
          </Paper>
        </Grid>

        {/* Recent Alerts - Compact */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              p: 1.5,
              maxHeight: 120,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
              overflow: 'auto'
            }}
          >
            <Typography
              variant="subtitle2"
              gutterBottom
              sx={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '0.85rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(33, 150, 243, 0.5)',
                mb: 1
              }}
            >
              <Warning sx={{ mr: 0.5, color: 'primary.main', fontSize: 16 }} />
              Recent Alerts
            </Typography>
            {alerts.length > 0 ? (
              <Box>
                {alerts.slice(0, 3).map((alert, index) => (
                  <Alert
                    key={index}
                    severity={alert.severity || 'info'}
                    sx={{ mb: 0.5, py: 0.5 }}
                  >
                    <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
                      <strong>{new Date(alert.timestamp).toLocaleTimeString()}:</strong> {alert.message || alert.attack_type || 'Threat detected'}
                    </Typography>
                  </Alert>
                ))}
              </Box>
            ) : (
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                No recent alerts
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default LiveDashboard;
