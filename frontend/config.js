/**
 * config.js — API base URL configuration.
 * Leave BACKEND_URL as empty string to use the same origin (recommended for Replit).
 * Change to a full URL (e.g. "https://my-api.onrender.com") for separate deployments.
 */
const CONFIG = {
    BACKEND_URL: "",       // Same-origin: FastAPI serves both frontend and API
    MAX_CHAR:    4000,     // Max characters per message
    VERSION:     "v2.0",
};
