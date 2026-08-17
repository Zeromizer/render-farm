import React from 'react';
import {Interactive} from 'remotion';

export const BrandBar: React.FC = () => (
  <Interactive.Div name="Proton e.MAS brand bar" style={{position: 'absolute', top: 64, left: 64, display: 'flex', alignItems: 'center', gap: 18, color: 'white', fontFamily: 'Arial, sans-serif', fontWeight: 900, letterSpacing: 2, fontSize: 30, zIndex: 50}}>
    <div style={{width: 34, height: 34, border: '5px solid white', borderRadius: 8, rotate: '45deg'}} />
    <span>PROTON</span><span style={{fontWeight: 500, opacity: 0.92}}>e.MAS</span>
  </Interactive.Div>
);
