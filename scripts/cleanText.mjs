export function cleanText(text) {
    return text
      .toLowerCase()                      // Convert to lowercase
      .replace(/[^a-zA-Z\s]/g, '')         // Remove punctuation
      .replace(/\s+/g, ' ')                // Remove extra spaces
      .trim();                             // Trim leading/trailing spaces
  }