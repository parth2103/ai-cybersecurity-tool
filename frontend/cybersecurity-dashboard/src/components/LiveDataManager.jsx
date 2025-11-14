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

  // Generate test features dynamically based on API requirements
  const generateTestFeatures = async (attackType = 'attack') => {
    // First, try to get a test data sample from the API
    let features = {};
    
    try {
      // Try to get a test data sample from API
      const testSampleResponse = await axios.get(`${API_URL}/test-data/sample`, {
        headers: { 'X-API-Key': API_KEY }
      });
      
      if (testSampleResponse.data.success && testSampleResponse.data.sample) {
        // Use the sample from API (has all required features)
        features = testSampleResponse.data.sample.features;
        console.log('✅ Using test data sample from API');
        return features;
      }
    } catch (e) {
      console.log('Could not get test data sample from API, generating default features');
    }
    
    // Fallback: Generate comprehensive feature set
    try {
      // Try to get system info
      const sysInfo = await axios.get(`${API_URL}/system/info`, {
        headers: { 'X-API-Key': API_KEY }
      });
    } catch (e) {
      console.log('Could not get system info');
    }
    
    // Generate comprehensive test features (covers both old and new feature sets)
    // This includes common features from CICIDS2017 and new datasets
    const baseFeatures = {
      // Core network features (common to both old and new)
      "Destination Port": attackType === 'attack' ? 80 : 443,
      "Flow Duration": attackType === 'attack' ? 1000 : 100000,
      "Total Fwd Packets": attackType === 'attack' ? 50000 : 10,
      "Total Backward Packets": attackType === 'attack' ? 0 : 10,
      "Total Length of Fwd Packets": attackType === 'attack' ? 5000000 : 2000,
      "Total Length of Bwd Packets": attackType === 'attack' ? 0 : 1800,
      "Fwd Packet Length Max": attackType === 'attack' ? 1500 : 300,
      "Fwd Packet Length Min": attackType === 'attack' ? 64 : 20,
      "Fwd Packet Length Mean": attackType === 'attack' ? 100 : 100,
      "Fwd Packet Length Std": attackType === 'attack' ? 0 : 50,
      "Bwd Packet Length Max": attackType === 'attack' ? 0 : 300,
      "Bwd Packet Length Min": attackType === 'attack' ? 0 : 20,
      "Bwd Packet Length Mean": attackType === 'attack' ? 0 : 100,
      "Bwd Packet Length Std": attackType === 'attack' ? 0 : 50,
      "Flow Bytes/s": attackType === 'attack' ? 100000000 : 2000,
      "Flow Packets/s": attackType === 'attack' ? 1000000 : 20,
      "Flow IAT Mean": attackType === 'attack' ? 0.05 : 50,
      "Flow IAT Std": attackType === 'attack' ? 0.01 : 10,
      "Flow IAT Max": attackType === 'attack' ? 0.1 : 100,
      "Flow IAT Min": attackType === 'attack' ? 0.01 : 10,
      "Fwd IAT Total": attackType === 'attack' ? 50 : 1000,
      "Fwd IAT Mean": attackType === 'attack' ? 0.001 : 10,
      "Fwd IAT Std": attackType === 'attack' ? 0.0001 : 5,
      "Fwd IAT Max": attackType === 'attack' ? 0.002 : 20,
      "Fwd IAT Min": attackType === 'attack' ? 0.0001 : 1,
      "Bwd IAT Total": attackType === 'attack' ? 0 : 1000,
      "Bwd IAT Mean": attackType === 'attack' ? 0 : 10,
      "Bwd IAT Std": attackType === 'attack' ? 0 : 5,
      "Bwd IAT Max": attackType === 'attack' ? 0 : 20,
      "Bwd IAT Min": attackType === 'attack' ? 0 : 1,
      "Fwd PSH Flags": 0,
      "Bwd PSH Flags": 0,
      "Fwd URG Flags": 0,
      "Bwd URG Flags": 0,
      "Fwd Header Length": 20,
      "Bwd Header Length": attackType === 'attack' ? 0 : 20,
      "Fwd Packets/s": attackType === 'attack' ? 1000000 : 20,
      "Bwd Packets/s": attackType === 'attack' ? 0 : 20,
      "Min Packet Length": 64,
      "Max Packet Length": 1500,
      "Packet Length Mean": attackType === 'attack' ? 100 : 100,
      "Packet Length Std": attackType === 'attack' ? 0 : 50,
      "Packet Length Variance": attackType === 'attack' ? 0 : 2500,
      "FIN Flag Count": 0,
      "SYN Flag Count": attackType === 'attack' ? 50000 : 1,
      "RST Flag Count": 0,
      "PSH Flag Count": 0,
      "ACK Flag Count": attackType === 'attack' ? 0 : 10,
      "URG Flag Count": 0,
      "CWE Flag Count": 0,
      "ECE Flag Count": 0,
      "Down/Up Ratio": attackType === 'attack' ? 0.002 : 1,
      "Average Packet Size": attackType === 'attack' ? 100 : 100,
      "Avg Fwd Segment Size": attackType === 'attack' ? 100 : 100,
      "Avg Bwd Segment Size": attackType === 'attack' ? 0 : 100,
      "Fwd Header Length.1": 20,
      "Fwd Avg Bytes/Bulk": 0,
      "Fwd Avg Packets/Bulk": 0,
      "Fwd Avg Bulk Rate": 0,
      "Bwd Avg Bytes/Bulk": 0,
      "Bwd Avg Packets/Bulk": 0,
      "Bwd Avg Bulk Rate": 0,
      "Subflow Fwd Packets": attackType === 'attack' ? 50000 : 10,
      "Subflow Fwd Bytes": attackType === 'attack' ? 5000000 : 2000,
      "Subflow Bwd Packets": attackType === 'attack' ? 0 : 10,
      "Subflow Bwd Bytes": attackType === 'attack' ? 0 : 1800,
      "Init_Win_bytes_forward": 65535,
      "Init_Win_bytes_backward": attackType === 'attack' ? 0 : 65535,
      "act_data_pkt_fwd": 0,
      "min_seg_size_forward": 0,
      "Active Mean": attackType === 'attack' ? 0 : 1000,
      "Active Std": 0,
      "Active Max": attackType === 'attack' ? 0 : 2000,
      "Active Min": 0,
      "Idle Mean": attackType === 'attack' ? 0 : 100,
      "Idle Std": 0,
      "Idle Max": attackType === 'attack' ? 0 : 200,
      "Idle Min": 0
    };
    
    // Add common new dataset features (if they exist in new feature set)
    // These will be filled by API if missing, but we include common ones
    const newFeatures = {
      "ACK Flag Count": attackType === 'attack' ? 0 : 10,
      "ARP": 0,
      "AVG": attackType === 'attack' ? 100 : 100,
      "Bwd Bulk Rate Avg": 0,
      "Bwd Bytes/Bulk Avg": 0,
      "Bwd Packets/Bulk Avg": 0,
    };
    
    // Merge all features
    features = { ...baseFeatures, ...newFeatures };
    
    // Fill any remaining common feature patterns with defaults
    // The API will handle missing features by filling with 0
    return features;
  };

  // Send test data to trigger alerts
  const sendTestData = async () => {
    console.log('🧪 Sending test data...');
    setIsTestSending(true);
    try {
      // Generate test features dynamically
      const testFeatures = await generateTestFeatures('attack');
      
      const testData = {
        features: testFeatures,
        source_ip: "192.168.1.100",
        attack_type: "DDoS_Test_New_Models"
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
