'use client';

import React from 'react';
import { ThemeProvider } from '@/components/landing/ThemeContext';
import Navbar from '@/components/landing/Navbar';
import Hero from '@/components/landing/Hero';
import Features from '@/components/landing/Features';
import HowItWorks from '@/components/landing/HowItWorks';
import Problem from '@/components/landing/Problem';
import UseCases from '@/components/landing/UseCases';
import CTA from '@/components/landing/CTA';
import Footer from '@/components/landing/Footer';
import Cursor from '@/components/landing/Cursor';

export default function LandingPage() {
  return (
    <ThemeProvider>
      <div className="landing-theme">
        <Cursor />
        <Navbar />
        <main>
          <Hero />
          <Problem />
          <Features />
          <HowItWorks />
          <UseCases />
          <CTA />
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}
