# Filters
Algorithm Implementation for statistical filters.

This repository contains:

1. Kalman Filter Implementation
2. Particle Filter Implementation
3. Examples of Filter Application


## Table of Contents
- [Background](##background)
- [Kalman Filter](#kalman-filter)
- [Particle Filter](#particle-filter)
- [Example](#example)
- [Contributions](#contributions)

## Background
State space models are widely used in many applications. Estimating the unknown state based on the available measurements is an important and a well studied subject in the literature. Depending on the nature of the problem, these models could involve simple linear equations or complex nonlinearities. Filtering is a general statistical approach for estimating the evolving unknown states from the noisy measurements.  

In this report, we consider the hidden Markov model, where inference is tractable conditional on the history of the state of the hidden component. We will take an overview to several filtering algorithms by theoretical derivation and coding implementation and apply them to linear gaussian and  nonlinear and non-Gaussian cases.

## Kalman Filter
- Kalman Filter
- Extended Kalman Filter
- Unscented Kalman Filter

## Particle Filter
- Sequential importance resampling particle filter
  1. Linear Gaussian Case
  2. Nonlinear Gaussian Case

- Sequential importance adaptive resampling particle filter
  1. Linear Gaussian Case
  2. Nonlinear Gaussian Case

- Bootstrap filter
  1. Linear Gaussian Case
  2. Nonlinear Gaussian Case

## Example
- Car tracking demo
- Pendulum tracking demo
- Cluttered pendulum tracking demo

## Contributions
Xiandong Zou: Particle Filters and their demos
Tony Dam: Kalman Filters and their demos
