package config

import "os"

type Config struct {
	Environment string
	Server      ServerConfig
	Azure       AzureConfig
	Database    DatabaseConfig
}

type ServerConfig struct {
	Port string
}

type AzureConfig struct {
	Endpoint string
	APIKey   string
	Region   string
	// GPT/OpenAI Configuration for validation
	GPTEndpoint   string
	GPTAPIKey     string
	GPTDeployment string
	GPTAPIVersion string
}

type DatabaseConfig struct {
	URL string
}

func New() *Config {
	return &Config{
		Environment: getEnv("NODE_ENV", "development"),
		Server: ServerConfig{
			Port: getEnv("PORT", "8080"),
		},
		Azure: AzureConfig{
			Endpoint:      getEnv("AZURE_ENDPOINT", ""),
			APIKey:        getEnv("AZURE_API_KEY", ""),
			Region:        getEnv("AZURE_REGION", "eastus"),
			GPTEndpoint:   getEnv("AZURE_GPT_ENDPOINT", ""),
			GPTAPIKey:     getEnv("AZURE_API_KEY", ""), // Same key for both services
			GPTDeployment: getEnv("AZURE_GPT_DEPLOYMENT", "gpt-4o-mini"),
			GPTAPIVersion: getEnv("AZURE_GPT_API_VERSION", "2024-08-01-preview"),
		},
		Database: DatabaseConfig{
			URL: getEnv("DATABASE_URL", ""),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
