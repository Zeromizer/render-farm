import React from 'react';
import {Img, Interactive, staticFile} from 'remotion';

export const BrandBar: React.FC = () => (
  <Interactive.Div name="Proton e.MAS brand bar" style={{position: 'absolute', top: 64, left: 64, display: 'flex', alignItems: 'center', gap: 18, color: 'white', fontFamily: 'Arial, sans-serif', fontWeight: 900, letterSpacing: 2, fontSize: 30, zIndex: 50}}>
    <Img name="Authentic Proton tiger-head emblem" src={staticFile('ending/proton-tiger-head-white.png')} style={{width: 42, height: 42, objectFit: 'contain'}} />
    <span>PROTON</span><span style={{fontWeight: 500, opacity: 0.92}}>e.MAS</span>
  </Interactive.Div>
);
