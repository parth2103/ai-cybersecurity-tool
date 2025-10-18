import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Container, Grid, Paper, Typography, Card, CardContent,
  Alert, Box, LinearProgress, Chip, Button
} from '@mui/material';
import {
  Security, Speed, Shield, TrendingUp, Storage, CloudQueue,
  Warning, CheckCircle, Error as ErrorIcon
} from '@mui/icons-material';
import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';
import AttentionVisualizer from './AttentionVisualizer';
import ModelPerformanceMonitor from './ModelPerformanceMonitor';
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
        background: `radial-gradient(circle, ${chartColors[color]}20 0%, transparent 70%)`,
        transform: 'translate(0, -50%)',
        opacity: 0.3,
      },
      '&:hover': {
        transform: 'translateY(-4px) scale(1.02)',
        boxShadow: `0px 12px 24px rgba(${color === 'error' ? '244, 67, 54' : color === 'warning' ? '255, 152, 0' : color === 'success' ? '76, 175, 80' : '33, 150, 243'}, 0.3)`,
        borderColor: chartColors[color],
        '&::before': {
          height: '4px',
        },
        '& .metric-icon': {
          transform: 'scale(1.2) rotate(5deg)',
        },
      },
      '@keyframes shimmer': {
        '0%, 100%': { opacity: 1 },
        '50%': { opacity: 0.5 },
      },
    }}
  >
    <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 }, position: 'relative', zIndex: 1 }}>
      <Box display="flex" alignItems="center" mb={0.5}>
        <Icon
          className="metric-icon"
          sx={{
            fontSize: 24,
            color: chartColors[color],
            mr: 1,
            opacity: 0.9,
            transition: 'all 0.3s ease',
            filter: `drop-shadow(0 0 8px ${chartColors[color]}60)`,
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', fontWeight: 500 }}>
          {title}
        </Typography>
      </Box>
      <Typography
        variant="h4"
        sx={{
          fontWeight: 700,
          color: chartColors[color],
          fontSize: '1.75rem',
          textShadow: `0 0 20px ${chartColors[color]}40`,
          transition: 'all 0.3s ease',
        }}
      >
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', opacity: 0.8 }}>
          {subtitle}
        </Typography>
      )}
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const [stats, setStats] = useState({
    total_requests: 0,
    threats_detected: 0,
    threat_history: [],
    current_threat_level: 'Low'
  });
  const [alerts, setAlerts] = useState([]);
  const [systemInfo, setSystemInfo] = useState({});
  const [loading, setLoading] = useState(true);
  const [realtimeData, setRealtimeData] = useState([]);
  const [apiConnectionStatus, setApiConnectionStatus] = useState('Connecting...');
  const [error, setError] = useState(null);
  const [testBusy, setTestBusy] = useState(false);
  const [explainFeatures, setExplainFeatures] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      setApiConnectionStatus('Fetching stats...');
      const response = await axios.get(`${API_URL}/stats`, { headers: { 'X-API-Key': API_KEY } });
      setStats(response.data);
      setApiConnectionStatus('Connected');
      setError(null);

      if (response.data.threat_history.length > 0) {
        const chartData = response.data.threat_history.slice(-20).map((item, index) => ({
          time: new Date(item.timestamp).toLocaleTimeString(),
          threat_score: item.threat_score * 100,
          index: index
        }));
        setRealtimeData(chartData);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setApiConnectionStatus('Connection failed');
      setError(`Failed to fetch stats: ${error.response?.data?.error || error.message}`);
      setLoading(false);
    }
  }, []);

  const fetchSystemInfo = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/system/info`, { headers: { 'X-API-Key': API_KEY } });
      setSystemInfo(response.data);
    } catch (error) {
      console.error('Error fetching system info:', error);
      setError(`Failed to fetch system info: ${error.response?.data?.error || error.message}`);
    }
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      const incoming = [];
      for (const th of stats.threat_history || []) {
        if (th.threat_score > 0.5) {
          incoming.push({
            timestamp: th.timestamp,
            threat_level: th.threat_level,
            threat_score: th.threat_score,
            source_ip: th.source_ip || 'Unknown',
            attack_type: th.attack_type || 'Unknown Attack'
          });
        }
      }
      setAlerts(incoming);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  }, [stats.threat_history]);


  useEffect(() => {
    // Initial load
    (async () => {
      try {
        await fetchStats();
        await fetchSystemInfo();
        await fetchAlerts();
      } catch (error) {
        console.error('Initial load error:', error);
      }
    })();

    // Enable automatic polling every 10 seconds
    const interval = setInterval(async () => {
      try {
        await fetchStats();
        await fetchAlerts();
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 10000);

    return () => {
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const getThreatLevelColor = (level) => {
    const colors = {
      'Critical': chartColors.error,
      'High': chartColors.warning,
      'Medium': '#ffc107',
      'Low': chartColors.success,
      'None': chartColors.info
    };
    return colors[level] || '#9e9e9e';
  };

  const getThreatIcon = (level) => {
    switch (level) {
      case 'Critical':
      case 'High':
        return <ErrorIcon />;
      case 'Medium':
        return <Warning />;
      default:
        return <CheckCircle />;
    }
  };

  const pieData = [
    { name: 'Safe', value: stats.total_requests - stats.threats_detected },
    { name: 'Threats', value: stats.threats_detected }
  ];

  const COLORS = [chartColors.success, chartColors.error];

  const detectionRate = stats.total_requests > 0
    ? ((stats.threats_detected / stats.total_requests) * 100).toFixed(2)
    : 0;

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
        }}
      >
        <Security sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          Loading Dashboard...
        </Typography>
        <Box sx={{ width: '300px', mt: 2 }}>
          <LinearProgress />
        </Box>
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      {/* Header Section */}
      <Box sx={{ mb: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Box display="flex" alignItems="center">
            <Security sx={{ fontSize: 36, color: 'primary.main', mr: 1.5 }} />
            <Typography
              variant="h4"
              sx={{
                background: 'linear-gradient(45deg, #2196f3 30%, #00bcd4 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontSize: '1.75rem',
              }}
            >
              AI Cybersecurity Dashboard
            </Typography>
          </Box>

          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              icon={apiConnectionStatus === 'Connected' ? <CheckCircle /> : <ErrorIcon />}
              label={apiConnectionStatus}
              color={apiConnectionStatus === 'Connected' ? 'success' : 'error'}
              size="small"
            />
            <Button
              variant="outlined"
              size="small"
              disabled={testBusy}
              onClick={async () => {
                if (testBusy) return;
                setTestBusy(true);
                setError(null);
                try {
                  await Promise.all([fetchStats(), fetchSystemInfo()]);
                } finally {
                  setTimeout(() => setTestBusy(false), 1000);
                }
              }}
              sx={{ minWidth: 120 }}
            >
              {testBusy ? 'Testing…' : 'TEST'}
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mt: 1, py: 0.5 }}>
            {error}
          </Alert>
        )}
      </Box>

      <Grid container spacing={2}>
        {/* Threat Level Alert Banner with Animation */}
        <Grid item xs={12}>
          <Alert
            severity={stats.current_threat_level === 'Critical' ? 'error' :
                    stats.current_threat_level === 'High' ? 'warning' :
                    stats.current_threat_level === 'Medium' ? 'warning' : 'success'}
            icon={getThreatIcon(stats.current_threat_level)}
            sx={{
              borderRadius: 2,
              borderLeft: `4px solid ${getThreatLevelColor(stats.current_threat_level)}`,
              py: 0.5,
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
              '& .MuiAlert-icon': {
                animation: stats.current_threat_level === 'Critical' || stats.current_threat_level === 'High'
                  ? 'iconPulse 1s ease-in-out infinite'
                  : 'none',
              },
              '@keyframes iconPulse': {
                '0%, 100%': { transform: 'scale(1)' },
                '50%': { transform: 'scale(1.15)' },
              },
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Current Threat Level: <strong>{stats.current_threat_level}</strong>
            </Typography>
          </Alert>
        </Grid>

        {/* Stats Cards - Grouped */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Requests"
            value={stats.total_requests.toLocaleString()}
            icon={Storage}
            color="info"
            subtitle="All processed requests"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Threats Detected"
            value={stats.threats_detected.toLocaleString()}
            icon={Warning}
            color="error"
            subtitle="Security incidents"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Detection Rate"
            value={`${detectionRate}%`}
            icon={TrendingUp}
            color="warning"
            subtitle="Threat identification rate"
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

        {/* Main Charts Section - Grouped with Glassmorphism */}
        <Grid item xs={12} md={8}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: 280,
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
                fontSize: '1rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(33, 150, 243, 0.5)',
              }}
            >
              <Speed sx={{ mr: 1, color: 'primary.main', fontSize: 20, filter: 'drop-shadow(0 0 6px rgba(33, 150, 243, 0.6))' }} />
              Real-time Threat Scores
            </Typography>
            <ResponsiveContainer width="100%" height={230}>
              <AreaChart data={realtimeData}>
                <defs>
                  <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartColors.error} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={chartColors.error} stopOpacity={0.1}/>
                  </linearGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                      <feMergeNode in="coloredBlur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} strokeOpacity={0.3} />
                <XAxis dataKey="time" stroke={chartColors.axis} style={{ fontSize: '0.7rem' }} />
                <YAxis domain={[0, 100]} stroke={chartColors.axis} style={{ fontSize: '0.7rem' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(30, 30, 30, 0.95)',
                    border: `1px solid ${chartColors.tooltip.border}`,
                    borderRadius: 8,
                    backdropFilter: 'blur(10px)',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="threat_score"
                  stroke={chartColors.error}
                  strokeWidth={3}
                  fill="url(#colorThreat)"
                  animationDuration={1000}
                  filter="url(#glow)"
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
              height: 280,
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(76, 175, 80, 0.3)',
              boxShadow: '0 8px 32px rgba(76, 175, 80, 0.15), inset 0 0 30px rgba(76, 175, 80, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'rgba(76, 175, 80, 0.5)',
                boxShadow: '0 12px 40px rgba(76, 175, 80, 0.25), inset 0 0 40px rgba(76, 175, 80, 0.08)',
              },
            }}
          >
            <Typography
              variant="subtitle1"
              gutterBottom
              sx={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '1rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(76, 175, 80, 0.5)',
              }}
            >
              <Shield sx={{ mr: 1, color: 'success.main', fontSize: 20, filter: 'drop-shadow(0 0 6px rgba(76, 175, 80, 0.6))' }} />
              Threat Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  animationDuration={1000}
                  style={{ fontSize: '0.7rem' }}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(30, 30, 30, 0.95)',
                    border: `1px solid ${chartColors.tooltip.border}`,
                    borderRadius: 8,
                    backdropFilter: 'blur(10px)',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Performance Section - Left Column */}
        <Grid item xs={12} md={5}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: '100%',
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              boxShadow: '0 8px 32px rgba(33, 150, 243, 0.15), inset 0 0 30px rgba(33, 150, 243, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'rgba(33, 150, 243, 0.5)',
                boxShadow: '0 12px 40px rgba(33, 150, 243, 0.25)',
              },
            }}
          >
            <ModelPerformanceMonitor
              apiBaseUrl={API_URL}
              apiKey={API_KEY}
            />
          </Paper>
        </Grid>

        {/* Feature Attention Visualizer - Middle Column */}
        <Grid item xs={12} md={4}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: '100%',
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 152, 0, 0.3)',
              boxShadow: '0 8px 32px rgba(255, 152, 0, 0.15), inset 0 0 30px rgba(255, 152, 0, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'rgba(255, 152, 0, 0.5)',
                boxShadow: '0 12px 40px rgba(255, 152, 0, 0.25)',
              },
            }}
          >
            <AttentionVisualizer
              apiBaseUrl={API_URL}
              apiKey={API_KEY}
              features={explainFeatures || stats.threat_history?.[stats.threat_history.length - 1]?.features}
              title="Feature Attention"
            />
          </Paper>
        </Grid>

        {/* Recent Alerts Section - Right Column */}
        <Grid item xs={12} md={3}>
          <Paper
            elevation={3}
            sx={{
              p: 2,
              height: '100%',
              background: 'linear-gradient(135deg, rgba(21, 26, 53, 0.8) 0%, rgba(26, 32, 64, 0.9) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(244, 67, 54, 0.3)',
              boxShadow: '0 8px 32px rgba(244, 67, 54, 0.15), inset 0 0 30px rgba(244, 67, 54, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'rgba(244, 67, 54, 0.5)',
                boxShadow: '0 12px 40px rgba(244, 67, 54, 0.25)',
              },
            }}
          >
            <Typography
              variant="subtitle1"
              gutterBottom
              sx={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '1rem',
                fontWeight: 600,
                textShadow: '0 0 10px rgba(244, 67, 54, 0.5)',
              }}
            >
              <Warning sx={{ mr: 1, color: 'warning.main', fontSize: 20, filter: 'drop-shadow(0 0 6px rgba(255, 152, 0, 0.6))' }} />
              Recent Alerts
            </Typography>
            <Box
              sx={{
                maxHeight: 380,
                overflow: 'auto',
                '&::-webkit-scrollbar': {
                  width: '6px',
                },
                '&::-webkit-scrollbar-track': {
                  background: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: 'rgba(33, 150, 243, 0.5)',
                  borderRadius: '4px',
                  '&:hover': {
                    background: 'rgba(33, 150, 243, 0.7)',
                  },
                },
              }}
            >
              {alerts.length === 0 ? (
                <Box textAlign="center" py={2}>
                  <CheckCircle sx={{ fontSize: 32, color: 'success.main', mb: 1 }} />
                  <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.85rem' }}>No recent alerts</Typography>
                </Box>
              ) : (
                alerts.map((alert, index) => (
                  <Alert
                    key={index}
                    severity={alert.threat_level === 'Critical' ? 'error' : 'warning'}
                    icon={getThreatIcon(alert.threat_level)}
                    sx={{ mb: 1, borderRadius: 2, py: 0.5 }}
                  >
                    <Box>
                      <Typography variant="caption" fontWeight={600} sx={{ fontSize: '0.75rem', display: 'block' }}>
                        {alert.attack_type || 'Unknown Attack'}
                      </Typography>
                      <Typography variant="caption" sx={{ fontSize: '0.65rem', display: 'block', mb: 0.5 }}>
                        {alert.source_ip}
                      </Typography>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Chip
                          label={alert.threat_level}
                          size="small"
                          sx={{
                            backgroundColor: getThreatLevelColor(alert.threat_level),
                            color: 'white',
                            fontWeight: 600,
                            height: 18,
                            fontSize: '0.6rem',
                          }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6rem' }}>
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </Typography>
                      </Box>
                    </Box>
                  </Alert>
                ))
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
