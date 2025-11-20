import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import axios from 'axios';

const API_URL = 'http://localhost:5001';
const API_KEY = 'dev-key-123';

const LiveDataManager = ({ onStatsUpdate, onAlertsUpdate, onSystemUpdate, onFeaturesReady }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [isTestSending, setIsTestSending] = useState(false);
  const [isContinuousSending, setIsContinuousSending] = useState(false);
  const [currentAttackIndex, setCurrentAttackIndex] = useState(0);
  const intervalRef = useRef(null);
  const continuousIntervalRef = useRef(null);
  const callbacksRef = useRef({ onStatsUpdate, onAlertsUpdate, onSystemUpdate, onFeaturesReady });
  
  // Attack types to rotate through
  const attackTypes = ['DDoS', 'DoS', 'Mirai Botnet', 'Brute Force', 'Reconnaissance', 'Spoofing', 'IIoT Attack', 'Benign'];

  // Update callbacks ref when they change
  useEffect(() => {
    callbacksRef.current = { onStatsUpdate, onAlertsUpdate, onSystemUpdate, onFeaturesReady };
  }, [onStatsUpdate, onAlertsUpdate, onSystemUpdate, onFeaturesReady]);

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

  // Generate test features dynamically based on API requirements and attack type
  const generateTestFeatures = async (attackTypeName = 'DDoS') => {
    // Map attack type names to feature generation logic
    const isAttack = attackTypeName !== 'Benign';
    const attackType = isAttack ? 'attack' : 'benign';
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
    // Adjust features based on specific attack type
    let port = 80;
    let fwdPackets = isAttack ? 50000 : 10;
    let bwdPackets = isAttack ? 0 : 10;
    let duration = isAttack ? 1000 : 100000;
    let bytesPerSec = isAttack ? 100000000 : 2000;
    let packetsPerSec = isAttack ? 1000000 : 20;
    
    // Adjust for specific attack types
    if (attackTypeName === 'DDoS') {
      port = 80;
      fwdPackets = 100000;
      bwdPackets = 0;
      duration = 50;
      bytesPerSec = 200000000;
      packetsPerSec = 2000000;
    } else if (attackTypeName === 'DoS') {
      port = 80;
      fwdPackets = 10000;
      bwdPackets = 100;
      duration = 500;
      bytesPerSec = 20000000;
      packetsPerSec = 200000;
    } else if (attackTypeName === 'Mirai Botnet') {
      port = 23; // Telnet
      fwdPackets = 5000;
      bwdPackets = 250;
      duration = 50000;
      bytesPerSec = 5000000;
      packetsPerSec = 100000;
    } else if (attackTypeName === 'Brute Force') {
      port = 22; // SSH
      fwdPackets = 50;
      bwdPackets = 50;
      duration = 2500;
      bytesPerSec = 5000;
      packetsPerSec = 50;
    } else if (attackTypeName === 'Reconnaissance') {
      port = 22; // Various ports scanned
      fwdPackets = 1;
      bwdPackets = 0;
      duration = 250;
      bytesPerSec = 100;
      packetsPerSec = 1;
    } else if (attackTypeName === 'Spoofing') {
      port = 53; // DNS
      fwdPackets = 500;
      bwdPackets = 0;
      duration = 5000;
      bytesPerSec = 50000;
      packetsPerSec = 500;
    } else if (attackTypeName === 'IIoT Attack') {
      port = 502; // Modbus
      fwdPackets = 5000;
      bwdPackets = 5000;
      duration = 25000;
      bytesPerSec = 25000000;
      packetsPerSec = 200000;
    } else if (attackTypeName === 'Benign') {
      port = 443;
      fwdPackets = 10;
      bwdPackets = 10;
      duration = 100000;
      bytesPerSec = 2000;
      packetsPerSec = 20;
    }
    
    const baseFeatures = {
      // Core network features (common to both old and new)
      "Destination Port": port,
      "Flow Duration": duration,
      "Total Fwd Packets": fwdPackets,
      "Total Backward Packets": bwdPackets,
      "Total Length of Fwd Packets": attackType === 'attack' ? 5000000 : 2000,
      "Total Length of Bwd Packets": attackType === 'attack' ? 0 : 1800,
      "Fwd Packet Length Max": attackType === 'attack' ? 1500 : 300,
      "Fwd Packet Length Min": attackType === 'attack' ? 64 : 20,
      "Fwd Packet Length Mean": attackType === 'attack' ? 100 : 100,
      "Fwd Packet Length Std": attackType === 'attack' ? 0 : 50,
      "Bwd Packet Length Max": isAttack ? 0 : 300,
      "Bwd Packet Length Min": isAttack ? 0 : 20,
      "Bwd Packet Length Mean": isAttack ? 0 : 100,
      "Bwd Packet Length Std": isAttack ? 0 : 50,
      "Flow Bytes/s": bytesPerSec,
      "Flow Packets/s": packetsPerSec,
      "Flow IAT Mean": isAttack ? 0.05 : 50,
      "Flow IAT Std": isAttack ? 0.01 : 10,
      "Flow IAT Max": isAttack ? 0.1 : 100,
      "Flow IAT Min": isAttack ? 0.01 : 10,
      "Fwd IAT Total": isAttack ? 50 : 1000,
      "Fwd IAT Mean": isAttack ? 0.001 : 10,
      "Fwd IAT Std": isAttack ? 0.0001 : 5,
      "Fwd IAT Max": isAttack ? 0.002 : 20,
      "Fwd IAT Min": isAttack ? 0.0001 : 1,
      "Bwd IAT Total": isAttack ? 0 : 1000,
      "Bwd IAT Mean": isAttack ? 0 : 10,
      "Bwd IAT Std": isAttack ? 0 : 5,
      "Bwd IAT Max": isAttack ? 0 : 20,
      "Bwd IAT Min": isAttack ? 0 : 1,
      "Fwd PSH Flags": 0,
      "Bwd PSH Flags": 0,
      "Fwd URG Flags": 0,
      "Bwd URG Flags": 0,
      "Fwd Header Length": 20,
      "Bwd Header Length": isAttack ? 0 : 20,
      "Fwd Packets/s": packetsPerSec,
      "Bwd Packets/s": isAttack ? 0 : packetsPerSec / 2,
      "Min Packet Length": 64,
      "Max Packet Length": 1500,
      "Packet Length Mean": 100,
      "Packet Length Std": isAttack ? 0 : 50,
      "Packet Length Variance": isAttack ? 0 : 2500,
      "FIN Flag Count": 0,
      "SYN Flag Count": isAttack ? fwdPackets : 1,
      "RST Flag Count": 0,
      "PSH Flag Count": 0,
      "ACK Flag Count": isAttack ? 0 : 10,
      "URG Flag Count": 0,
      "CWE Flag Count": 0,
      "ECE Flag Count": 0,
      "Down/Up Ratio": isAttack ? (bwdPackets > 0 ? fwdPackets / bwdPackets : 0.002) : 1,
      "Average Packet Size": 100,
      "Avg Fwd Segment Size": 100,
      "Avg Bwd Segment Size": isAttack ? 0 : 100,
      "Fwd Header Length.1": 20,
      "Fwd Avg Bytes/Bulk": 0,
      "Fwd Avg Packets/Bulk": 0,
      "Fwd Avg Bulk Rate": 0,
      "Bwd Avg Bytes/Bulk": 0,
      "Bwd Avg Packets/Bulk": 0,
      "Bwd Avg Bulk Rate": 0,
      "Subflow Fwd Packets": fwdPackets,
      "Subflow Fwd Bytes": fwdPackets * 100,
      "Subflow Bwd Packets": bwdPackets,
      "Subflow Bwd Bytes": bwdPackets * 100,
      "Init_Win_bytes_forward": 65535,
      "Init_Win_bytes_backward": isAttack ? 0 : 65535,
      "act_data_pkt_fwd": 0,
      "min_seg_size_forward": 0,
      "Active Mean": isAttack ? 0 : 1000,
      "Active Std": 0,
      "Active Max": isAttack ? 0 : 2000,
      "Active Min": 0,
      "Idle Mean": isAttack ? 0 : 100,
      "Idle Std": 0,
      "Idle Max": isAttack ? 0 : 200,
      "Idle Min": 0
    };
    
    // Add common new dataset features (if they exist in new feature set)
    // These will be filled by API if missing, but we include common ones
    const newFeatures = {
      "ACK Flag Count": isAttack ? 0 : 10,
      "ARP": 0,
      "AVG": 100,
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
  const sendTestData = async (attackTypeName = null) => {
    console.log('🧪 Sending test data...');
    setIsTestSending(true);
    try {
      // Use provided attack type or get current from rotation
      const attackType = attackTypeName || attackTypes[currentAttackIndex];
      
      // Generate test features dynamically based on attack type
      const testFeatures = await generateTestFeatures(attackType);
      
      // Notify parent component about features for explanation
      if (callbacksRef.current.onFeaturesReady) {
        callbacksRef.current.onFeaturesReady(testFeatures);
      }
      
      const testData = {
        features: testFeatures,
        source_ip: `192.168.1.${Math.floor(Math.random() * 254) + 1}`,
        attack_type: attackType
      };

      const response = await axios.post(`${API_URL}/predict`, testData, {
        headers: { 
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        }
      });

      console.log(`🧪 ${attackType} test data sent:`, response.data);
      console.log('✅ Test completed - Threat Score:', response.data.threat_score, 'Level:', response.data.threat_level);
      
      // Rotate to next attack type for next send
      if (!attackTypeName) {
        setCurrentAttackIndex((prev) => (prev + 1) % attackTypes.length);
      }
      
      setIsTestSending(false);
      return { response: response.data, features: testFeatures };
    } catch (error) {
      console.error('❌ Error sending test data:', error);
      setIsTestSending(false);
      throw error;
    }
  };

  // Toggle continuous sending
  const toggleContinuousSending = () => {
    if (isContinuousSending) {
      // Stop continuous sending
      if (continuousIntervalRef.current) {
        clearInterval(continuousIntervalRef.current);
        continuousIntervalRef.current = null;
      }
      setIsContinuousSending(false);
      console.log('⏹️ Continuous sending stopped');
    } else {
      // Start continuous sending
      setIsContinuousSending(true);
      console.log('▶️ Continuous sending started');
      
      // Send first one immediately
      sendTestData();
      
      // Then send every 3-5 seconds
      continuousIntervalRef.current = setInterval(() => {
        sendTestData();
      }, 3000 + Math.random() * 2000); // Random between 3-5 seconds
    }
  };

  // Cleanup continuous sending on unmount
  useEffect(() => {
    return () => {
      if (continuousIntervalRef.current) {
        clearInterval(continuousIntervalRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    connectionStatus,
    socket,
    sendTestData,
    isTestSending,
    toggleContinuousSending,
    isContinuousSending
  };
};

export default LiveDataManager;
