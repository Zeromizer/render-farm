import React from 'react';
import {Img, Interactive, staticFile} from 'remotion';

export const BrandBar: React.FC = () => (
  <Interactive.Div name="Official Proton e.MAS brand bar" style={{position: 'absolute', top: 52, left: 58, zIndex: 50}}>
    <Img name="Official Proton e.MAS white lockup" src={staticFile('ending/proton-emas-official-white-v9.png')} style={{display: 'block', width: 330, height: 'auto'}} />
  </Interactive.Div>
);
