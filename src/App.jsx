import { useState, useCallback } from 'react'
import './App.css'
import Ortho from './Ortho'

function App() {
  // const [rotate, setRotate] = useState(false)
  // const [angle, setAngle] = useState(0)
  // const handleRotate = () => setRotate(r => !r)
  // const handleAngleChange = (v) => setAngle(v)
  const [remountKey, setRemountKey] = useState(0);

  // Handle file loading with cleanup
  const handleFileLoaded = useCallback(() => {
    console.log('File loaded, forcing remount');
    // Unmount first
    setRemountKey(-1);
    // Then remount after a short delay
    setTimeout(() => setRemountKey(Date.now()), 100);
  }, []);

  return (
    <div style={{ width: '100%', height: '100%'}}>
      <Ortho 
        // rotate={rotate} 
        // angle={angle} 
        key={remountKey} 
        onFileLoaded={handleFileLoaded}
      />
    </div>
  )
}

export default App
