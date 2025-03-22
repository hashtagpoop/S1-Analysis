-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the vectordb database if it doesn't exist
CREATE DATABASE vectordb;

-- Connect to vectordb and create extension there as well
\c vectordb;
CREATE EXTENSION IF NOT EXISTS vector; 