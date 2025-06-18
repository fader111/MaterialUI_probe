import * as React from 'react'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import Menu from '@mui/material/Menu'
import MenuItem from '@mui/material/MenuItem'
import Button from '@mui/material/Button'
import Divider from '@mui/material/Divider'
import RotateLeftIcon from '@mui/icons-material/RotateLeft'
import ShuffleIcon from '@mui/icons-material/Shuffle'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import SaveAltIcon from '@mui/icons-material/SaveAlt'
import PublishIcon from '@mui/icons-material/Publish'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import Slider from '@mui/material/Slider'

function MenuBar() {
  const [anchorEls, setAnchorEls] = React.useState({})
  const menus = [
    { label: 'File', items: ['New', 'Open', 'Save', 'Exit'] },
    { label: 'Options', items: ['Preferences', 'Settings'] },
    { label: 'Tools', items: ['Customize', 'Extensions'] },
    { label: 'About', items: ['About App', 'Help'] },
  ]
  const handleMenuOpen = (event, label) => {
    setAnchorEls((prev) => ({ ...prev, [label]: event.currentTarget }))
  }
  const handleMenuClose = (label) => {
    setAnchorEls((prev) => ({ ...prev, [label]: null }))
  }
  // Only one menu open at a time
  React.useEffect(() => {
    const openMenus = Object.entries(anchorEls).filter(([_, v]) => !!v)
    if (openMenus.length > 1) {
      // Close all but the last opened
      const last = openMenus[openMenus.length - 1][0]
      setAnchorEls((prev) => Object.fromEntries([[last, prev[last]]]))
    }
  }, [anchorEls])
  return (
    <AppBar position="static" color="default" elevation={0} sx={{ pointerEvents: 'auto', boxShadow: 'none', bgcolor: 'rgba(255,255,255,0.5)' }}>
      <Toolbar variant="dense" sx={{ pointerEvents: 'auto', justifyContent: 'center' }}>
        {menus.map((menu) => (
          <Box key={menu.label} sx={{ mr: 2, pointerEvents: 'auto' }}>
            <Typography
              aria-controls={anchorEls[menu.label] ? `${menu.label}-menu` : undefined}
              aria-haspopup="true"
              aria-expanded={Boolean(anchorEls[menu.label]) ? 'true' : undefined}
              onClick={(e) => handleMenuOpen(e, menu.label)}
              sx={{ cursor: 'pointer', fontWeight: 500, display: 'inline-block', pointerEvents: 'auto' }}
              variant="body1"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') handleMenuOpen(e, menu.label)
              }}
            >
              {menu.label}
            </Typography>
            <Menu
              id={`${menu.label}-menu`}
              anchorEl={anchorEls[menu.label]}
              open={Boolean(anchorEls[menu.label])}
              onClose={() => handleMenuClose(menu.label)}
              MenuListProps={{ 'aria-labelledby': `${menu.label}-button`, sx: { pointerEvents: 'auto' } }}
              disableEnforceFocus
              disableAutoFocusItem
              disablePortal
              PaperProps={{ sx: { pointerEvents: 'auto' } }}
            >
              {menu.items.map((item) => (
                <MenuItem key={item} onClick={() => handleMenuClose(menu.label)} sx={{ pointerEvents: 'auto' }}>
                  {item}
                </MenuItem>
              ))}
            </Menu>
          </Box>
        ))}
      </Toolbar>
    </AppBar>
  )
}

function LeftPanel({ onRotate }) {
  return (
    <Box sx={{ width: 120, bgcolor: 'rgba(245,245,245,0.5)', height: '100%', p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="subtitle1">Left Tools</Typography>
      <Button variant="contained" size="small" onClick={onRotate} title="Rotate the cube">
        <RotateLeftIcon />
      </Button>
      <Button variant="outlined" size="small" title="Shuffle (not implemented)">
        <ShuffleIcon />
      </Button>
      <Button variant="text" size="small" title="Reset (not implemented)">
        <RestartAltIcon />
      </Button>
    </Box>
  )
}

function RightPanel() {
  return (
    <Box sx={{ width: 120, bgcolor: 'rgba(245,245,245,0.5)', height: '100%', p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="subtitle1">Right Tools</Typography>
      <Button variant="contained" size="small" title="Export (not implemented)">
        <SaveAltIcon />
      </Button>
      <Button variant="outlined" size="small" title="Import (not implemented)">
        <PublishIcon />
      </Button>
      <Button variant="text" size="small" title="Info (not implemented)">
        <InfoOutlinedIcon />
      </Button>
    </Box>
  )
}

export default function Overlay({ children, onRotate, angle, onAngleChange }) {
  return (
    <Box sx={{ height: '100vh', width: '100vw', position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 1, overflow: 'hidden' }}>
      {/* Main content (scene) in the background */}
      <Box sx={{ position: 'absolute', inset: 0, zIndex: 0, width: '100vw', height: '100vh', pointerEvents: 'auto' }}>
        {children}
      </Box>
      {/* Overlay UI panels */}
      <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 2, pointerEvents: 'auto' }}>
        <MenuBar />
      </Box>
      <Box sx={{ position: 'absolute', top: 64, left: 0, bottom: 0, zIndex: 2, display: 'flex', flexDirection: 'column', pointerEvents: 'none' }}>
        <Box sx={{ pointerEvents: 'auto' }}><LeftPanel onRotate={onRotate} /></Box>
      </Box>
      <Box sx={{ position: 'absolute', top: 64, right: 0, bottom: 0, zIndex: 2, display: 'flex', flexDirection: 'column', pointerEvents: 'none' }}>
        <Box sx={{ pointerEvents: 'auto' }}><RightPanel /></Box>
      </Box>
      {/* Bottom slider */}
      <Box sx={{ position: 'absolute', left: 0, right: 0, bottom: 16, zIndex: 2, pointerEvents: 'none', display: 'flex', justifyContent: 'center' }}>
        <Box sx={{ width: 300, pointerEvents: 'auto', borderRadius: 2, px: 2, py: 1 }}>
          <Slider
            value={angle}
            min={0}
            max={360}
            step={1}
            onChange={(_, v) => onAngleChange(v)}
            valueLabelDisplay="auto"
            aria-label="Cube Angulation"
          />
        </Box>
      </Box>
    </Box>
  )
}
