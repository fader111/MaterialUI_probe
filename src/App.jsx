import { useState, useCallback } from 'react'
import './App.css'
import Ortho from './Ortho'

function App() {
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
        key={remountKey} 
        onFileLoaded={handleFileLoaded}
      />
    </div>
  )
}

export default App
