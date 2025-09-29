import React, { useState, useEffect } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import {
  Container, Grid, Paper, Typography, Card, CardContent,
  Alert, Box, LinearProgress, Chip, Button
} from '@mui/material';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

const API_URL = 'http://localhost:5001';
const API_KEY = 'dev-key-123';
const socket = io(API_URL);

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
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStats();
    fetchSystemInfo();
    fetchAlerts();

    socket.on('new_alert', (alert) => {
      setAlerts(prev => [alert, ...prev].slice(0, 20));
    });

    const interval = setInterval(() => {
      fetchStats();
      fetchSystemInfo();
    }, 5000);

    return () => {
      clearInterval(interval);
      socket.disconnect();
    };
  }, []);

  const fetchStats = async () => {
    try {
      setConnectionStatus('Fetching stats...');
      const response = await axios.get(`${API_URL}/stats`, { headers: { 'X-API-Key': API_KEY } });
      setStats(response.data);
      setConnectionStatus('Connected');
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
      setConnectionStatus('Connection failed');
      setError(`Failed to fetch stats: ${error.response?.data?.error || error.message}`);
    }
  };

  const fetchSystemInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/system/info`, { headers: { 'X-API-Key': 'admin-key-789' } });
      setSystemInfo(response.data);
    } catch (error) {
      console.error('Error fetching system info:', error);
      setError(`Failed to fetch system info: ${error.response?.data?.error || error.message}`);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_URL}/alerts`, { headers: { 'X-API-Key': API_KEY } });
      let incoming = response.data.alerts || [];
      if (incoming.length === 0 && stats.threat_history?.length) {
        const derived = stats.threat_history
          .slice(-50)
          .filter(h => (h.threat_level || 'None') !== 'None' && (h.threat_level !== 'Low'))
          .reverse()
          .map(h => ({
            timestamp: h.timestamp,
            threat_level: h.threat_level,
            threat_score: h.threat_score,
            source_ip: h.source_ip || 'Unknown',
            attack_type: h.attack_type || 'Detected Threat'
          }));
        incoming = derived.slice(0, 20);
      }
      setAlerts(incoming);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const getThreatLevelColor = (level) => {
    const colors = {
      'Critical': '#f44336',
      'High': '#ff9800',
      'Medium': '#ffc107',
      'Low': '#4caf50',
      'None': '#2196f3'
    };
    return colors[level] || '#9e9e9e';
  };

  const pieData = [
    { name: 'Safe', value: stats.total_requests - stats.threats_detected },
    { name: 'Threats', value: stats.threats_detected }
  ];

  const COLORS = ['#4caf50', '#f44336'];

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 4 }}>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" gutterBottom>
        AI Cybersecurity Dashboard
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Chip 
          label={`Status: ${connectionStatus}`} 
          color={connectionStatus === 'Connected' ? 'success' : 'error'}
          variant="outlined"
        />
        <Button 
          variant="outlined" 
          size="small" 
          onClick={() => { setError(null); fetchStats(); fetchSystemInfo(); }}
          sx={{ ml: 2 }}
        >
          Test Connection
        </Button>
        {error && (
          <Alert severity="error" sx={{ mt: 1 }}>
            {error}
          </Alert>
        )}
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Alert 
            severity={stats.current_threat_level === 'Critical' ? 'error' : 
                    stats.current_threat_level === 'High' ? 'warning' : 'info'}
            sx={{ mb: 2 }}
          >
            Current Threat Level: <strong>{stats.current_threat_level}</strong>
          </Alert>
        </Grid>

        {/* Stats Cards (uniform height) */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: 140, display: 'flex', alignItems: 'center' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Requests
              </Typography>
              <Typography variant="h4">
                {stats.total_requests.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: 140, display: 'flex', alignItems: 'center' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Threats Detected
              </Typography>
              <Typography variant="h4" color="error">
                {stats.threats_detected.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: 140, display: 'flex', alignItems: 'center' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Detection Rate
              </Typography>
              <Typography variant="h4">
                {systemInfo.detection_rate?.toFixed(2) || 0}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: 140, display: 'flex', alignItems: 'center' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Health
              </Typography>
              <Box display="flex" alignItems="center">
                <Box width="100%" mr={1}>
                  <LinearProgress 
                    variant="determinate" 
                    value={100 - (systemInfo.cpu_percent || 0)} 
                    color="success"
                  />
                </Box>
                <Box minWidth={35}>
                  <Typography variant="body2">
                    {(100 - (systemInfo.cpu_percent || 0)).toFixed(0)}%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Real-time Threat Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, minHeight: 380, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Real-time Threat Scores
            </Typography>
            <Box sx={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={realtimeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Area 
                    type="monotone" 
                    dataKey="threat_score" 
                    stroke="#ff5722" 
                    fill="#ff5722" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Threat Distribution */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, minHeight: 380, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Threat Distribution
            </Typography>
            <Box sx={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height={300}>
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
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, minHeight: 380, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Recent Alerts
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
              {alerts.length === 0 ? (
                <Typography color="textSecondary">No recent alerts</Typography>
              ) : (
                alerts.map((alert, index) => (
                  <Alert 
                    key={index}
                    severity={alert.threat_level === 'Critical' ? 'error' : 'warning'}
                    sx={{ mb: 1 }}
                  >
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="subtitle2">
                          {alert.attack_type || 'Unknown Attack'}
                        </Typography>
                        <Typography variant="body2">
                          Source: {alert.source_ip} | Score: {(alert.threat_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Box>
                        <Chip 
                          label={alert.threat_level}
                          size="small"
                          style={{ 
                            backgroundColor: getThreatLevelColor(alert.threat_level),
                            color: 'white'
                          }}
                        />
                        <Typography variant="caption" sx={{ ml: 2 }}>
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
