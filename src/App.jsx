import { useState } from 'react'
import './App.css'
import Overlay from './Overlay'
import Ortho from './Ortho'

function App() {
  const [rotate, setRotate] = useState(false)
  const [angle, setAngle] = useState(0)
  const handleRotate = () => setRotate(r => !r)
  const handleAngleChange = (v) => setAngle(v)

  return (
    <Overlay onRotate={handleRotate} angle={angle} onAngleChange={handleAngleChange}>
      <div style={{ width: '100%', height: '100%'}}>
        <Ortho rotate={rotate} angle={angle} />
      </div>
    </Overlay>
  )
}

export default App
