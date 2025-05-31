export function cleanText(text) {
  return text
    .toLowerCase()                     // Convert to lowercase
    .replace(/[^a-zA-Z\s]/g, ' ')      // Replace punctuation with a space
    .replace(/\s+/g, ' ')              // Collapse any sequence of spaces down to one
    .trim();                           // Trim leading/trailing spaces
}
