import { useState, useCallback } from 'react'
import './App.css'
import Ortho from './Ortho'


function App() {
  const [orthoData, setOrthoData] = useState(null);
  const [isFileLoaded, setIsFileLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [baseCaseFilename, setBaseCaseFilename] = useState(() => {
    try {
      const stored = localStorage.getItem('baseCaseFilename');
      return stored ? stored : null;
    } catch (e) {
      return null;
    }
  });

  // Handler to reload orthoData after file upload/processing
  const handleFileLoaded = useCallback(async (filename) => {
    setLoading(true);
    setIsFileLoaded(false);
    if (!filename) {
      console.error('handleFileLoaded called WITHOUT a filename! Using fallback baseCaseFilename from state or default.');
      filename = baseCaseFilename ? baseCaseFilename + '.oas' : '00000000.oas';
    }
    console.log('handleFileLoaded!!! called with filename:', filename);
    try {
      const base_case_id = filename.replace(/\.oas$/i, '');
      const response = await fetch("http://localhost:8000/get_case_data/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_case_id })
      });
      if (!response.ok) throw new Error('Failed to reload orthoData');
      const data = await response.json();
      setOrthoData(data);
      setBaseCaseFilename(base_case_id);
      setIsFileLoaded(true);
      setLoading(false);
      try {
        localStorage.setItem('baseCaseFilename', base_case_id);
      } catch (e) {
        console.warn('WARNING: Failed to save baseCaseFilename to localStorage', e);
      }
    } catch (err) {
      setLoading(false);
      setIsFileLoaded(false);
      console.error('Failed to reload orthoData:', err);
    }
  }, [baseCaseFilename]);

  return (
    <div style={{ width: '100%', height: '100%'}}>
      <Ortho
        orthoData={orthoData}
        setOrthoData={setOrthoData}
        isFileLoaded={isFileLoaded}
        loading={loading}
        onFileLoaded={handleFileLoaded}
        baseCaseFilename={baseCaseFilename}
      />
    </div>
  );
}

export default App
