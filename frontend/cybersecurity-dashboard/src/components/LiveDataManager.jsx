import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import axios from 'axios';

const API_URL = 'http://localhost:5001';
const API_KEY = 'dev-key-123';

const LiveDataManager = ({ onStatsUpdate, onAlertsUpdate, onSystemUpdate }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [isTestSending, setIsTestSending] = useState(false);
  const intervalRef = useRef(null);
  const callbacksRef = useRef({ onStatsUpdate, onAlertsUpdate, onSystemUpdate });

  // Update callbacks ref when they change
  useEffect(() => {
    callbacksRef.current = { onStatsUpdate, onAlertsUpdate, onSystemUpdate };
  }, [onStatsUpdate, onAlertsUpdate, onSystemUpdate]);

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io(API_URL, {
      transports: ['websocket', 'polling']
    });

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('🔗 WebSocket connected');
      setIsConnected(true);
      setConnectionStatus('Connected');
    });

    newSocket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected');
      setIsConnected(false);
      setConnectionStatus('Disconnected');
    });

    newSocket.on('connect_error', (error) => {
      console.error('❌ WebSocket connection error:', error);
      setIsConnected(false);
      setConnectionStatus('Connection Error');
    });

    // Listen for real-time alerts
    newSocket.on('new_alert', (alert) => {
      console.log('🚨 New alert received:', alert);
      if (callbacksRef.current.onAlertsUpdate) {
        callbacksRef.current.onAlertsUpdate(alert);
      }
    });

    setSocket(newSocket);

    // Fallback: Set up polling if WebSocket fails
    const setupPolling = () => {
      const fetchData = async () => {
        try {
          // Fetch stats
          const statsResponse = await axios.get(`${API_URL}/stats`, {
            headers: { 'X-API-Key': API_KEY }
          });
          if (callbacksRef.current.onStatsUpdate) {
            callbacksRef.current.onStatsUpdate(statsResponse.data);
          }

          // Fetch system info
          const systemResponse = await axios.get(`${API_URL}/system/info`, {
            headers: { 'X-API-Key': 'admin-key-789' }
          });
          if (callbacksRef.current.onSystemUpdate) {
            callbacksRef.current.onSystemUpdate(systemResponse.data);
          }
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };

      // Initial fetch
      fetchData();

      // Set up polling every 2 seconds for live updates
      intervalRef.current = setInterval(fetchData, 2000);
    };

    // Use WebSocket if available, otherwise fallback to polling
    if (newSocket.connected) {
      // WebSocket is connected, use it for alerts
      console.log('✅ Using WebSocket for real-time updates');
    } else {
      // Fallback to polling
      console.log('🔄 Using polling for data updates');
      setupPolling();
    }

    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      newSocket.close();
    };
  }, []); // Empty dependency array to prevent infinite loops

  // Send test data to trigger alerts
  const sendTestData = async () => {
    console.log('🧪 Sending test data...');
    setIsTestSending(true);
    try {
      const testData = {
        features: {
          "Destination Port": 80,
          "Flow Duration": 50,
          "Total Fwd Packets": 50000,
          "Total Backward Packets": 100,
          "Total Length of Fwd Packets": 5000000,
          "Total Length of Bwd Packets": 10000,
          "Fwd Packet Length Max": 1500,
          "Fwd Packet Length Min": 64,
          "Fwd Packet Length Mean": 100,
          "Fwd Packet Length Std": 200,
          "Bwd Packet Length Max": 100,
          "Bwd Packet Length Min": 0,
          "Bwd Packet Length Mean": 10,
          "Bwd Packet Length Std": 20,
          "Flow Bytes/s": 100000000,
          "Flow Packets/s": 1000000,
          "Flow IAT Mean": 0.05,
          "Flow IAT Std": 0.01,
          "Flow IAT Max": 0.1,
          "Flow IAT Min": 0.01,
          "Fwd IAT Total": 50,
          "Fwd IAT Mean": 0.001,
          "Fwd IAT Std": 0.0001,
          "Fwd IAT Max": 0.002,
          "Fwd IAT Min": 0.0001,
          "Bwd IAT Total": 100,
          "Bwd IAT Mean": 1,
          "Bwd IAT Std": 0.5,
          "Bwd IAT Max": 2,
          "Bwd IAT Min": 0.5,
          "Fwd PSH Flags": 0,
          "Bwd PSH Flags": 0,
          "Fwd URG Flags": 0,
          "Bwd URG Flags": 0,
          "Fwd Header Length": 20,
          "Bwd Header Length": 20,
          "Fwd Packets/s": 1000000,
          "Bwd Packets/s": 1000,
          "Min Packet Length": 64,
          "Max Packet Length": 1500,
          "Packet Length Mean": 100,
          "Packet Length Std": 200,
          "Packet Length Variance": 40000,
          "FIN Flag Count": 0,
          "SYN Flag Count": 50000,
          "RST Flag Count": 0,
          "PSH Flag Count": 0,
          "ACK Flag Count": 100,
          "URG Flag Count": 0,
          "CWE Flag Count": 0,
          "ECE Flag Count": 0,
          "Down/Up Ratio": 0.002,
          "Average Packet Size": 100,
          "Avg Fwd Segment Size": 100,
          "Avg Bwd Segment Size": 10,
          "Fwd Header Length.1": 20,
          "Fwd Avg Bytes/Bulk": 0,
          "Fwd Avg Packets/Bulk": 0,
          "Fwd Avg Bulk Rate": 0,
          "Bwd Avg Bytes/Bulk": 0,
          "Bwd Avg Packets/Bulk": 0,
          "Bwd Avg Bulk Rate": 0,
          "Subflow Fwd Packets": 50000,
          "Subflow Fwd Bytes": 5000000,
          "Subflow Bwd Packets": 100,
          "Subflow Bwd Bytes": 10000,
          "Init_Win_bytes_forward": 65535,
          "Init_Win_bytes_backward": 0,
          "act_data_pkt_fwd": 0,
          "min_seg_size_forward": 0,
          "Active Mean": 0,
          "Active Std": 0,
          "Active Max": 0,
          "Active Min": 0,
          "Idle Mean": 0,
          "Idle Std": 0,
          "Idle Max": 0,
          "Idle Min": 0
        },
        source_ip: "192.168.1.100",
        attack_type: "DDoS_Test"
      };

      const response = await axios.post(`${API_URL}/predict`, testData, {
        headers: { 
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        }
      });

      console.log('🧪 Test data sent:', response.data);
      console.log('✅ Test completed - Threat Score:', response.data.threat_score, 'Level:', response.data.threat_level);
      setIsTestSending(false);
      return response.data;
    } catch (error) {
      console.error('❌ Error sending test data:', error);
      setIsTestSending(false);
      throw error;
    }
  };

  return {
    isConnected,
    connectionStatus,
    socket,
    sendTestData,
    isTestSending
  };
};

export default LiveDataManager;
