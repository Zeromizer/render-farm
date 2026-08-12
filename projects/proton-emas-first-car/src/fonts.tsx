import React from 'react';
import {staticFile} from 'remotion';

export const DISPLAY_FONT = 'Barlow Condensed, Arial Narrow, sans-serif';
export const BODY_FONT = 'Barlow, Arial, sans-serif';

export const FontStyles: React.FC = () => <style>{`
  @font-face { font-family: 'Barlow'; src: url('${staticFile('fonts/Barlow-Regular.ttf')}') format('truetype'); font-weight: 400; }
  @font-face { font-family: 'Barlow Condensed'; src: url('${staticFile('fonts/BarlowCondensed-Bold.ttf')}') format('truetype'); font-weight: 700; }
`}</style>;
