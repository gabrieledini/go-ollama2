package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
)

// Config struttura per le configurazioni
type Config struct {
	OllamaURL    string
	Model        string
	ChunkSize    int
	ChunkOverlap int
	TopK         int
	DataFile     string
}

// DocumentChunk rappresenta un pezzo di documento
type DocumentChunk struct {
	Text       string    `json:"text"`
	Embedding  []float32 `json:"embedding"`
	Similarity float32   `json:"similarity,omitempty"`
}

// KnowledgeBase gestisce la base di conoscenza
type KnowledgeBase struct {
	Chunks     []DocumentChunk `json:"chunks"`
	Embeddings embeddings.Embedder
	config     *Config
}

// OllamaEmbedder implementa l'interfaccia Embedder per Ollama
type OllamaEmbedder struct {
	BaseURL string
	Model   string
}

// EmbedDocuments implementa l'embedding per documenti
func (oe *OllamaEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))

	for i, text := range texts {
		embedding, err := oe.embedText(text)
		if err != nil {
			return nil, fmt.Errorf("error embedding text %d: %w", i, err)
		}
		results[i] = embedding
	}

	return results, nil
}

// EmbedQuery implementa l'embedding per query
func (oe *OllamaEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return oe.embedText(text)
}

// embedText esegue l'embedding di un singolo testo usando Ollama
func (oe *OllamaEmbedder) embedText(text string) ([]float32, error) {
	// Payload per Ollama embeddings API
	payload := map[string]interface{}{
		"model":  oe.Model,
		"prompt": text,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(oe.BaseURL+"/api/embeddings", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Embedding []float32 `json:"embedding"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Embedding, nil
}

// NewKnowledgeBase crea una nuova base di conoscenza
func NewKnowledgeBase(config *Config) *KnowledgeBase {
	embedder := &OllamaEmbedder{
		BaseURL: config.OllamaURL,
		Model:   "nomic-embed-text", // Modello di embedding locale
	}

	return &KnowledgeBase{
		Chunks:     make([]DocumentChunk, 0),
		Embeddings: embedder,
		config:     config,
	}
}

// LoadFromPDF carica e processa un file PDF
func (kb *KnowledgeBase) LoadFromPDF(pdfPath string) error {
	fmt.Printf("📄 Caricamento PDF: %s\n", pdfPath)

	// Estrai testo dal PDF
	text, err := extractTextFromPDF(pdfPath)
	if err != nil {
		return fmt.Errorf("errore nell'estrazione PDF: %w", err)
	}

	fmt.Printf("📝 Estratti %d caratteri dal PDF\n", len(text))

	// Dividi in chunks
	chunks := splitTextIntoChunks(text, kb.config.ChunkSize, kb.config.ChunkOverlap)
	fmt.Printf("🔢 Creati %d chunks di testo\n", len(chunks))

	// Crea embeddings
	fmt.Println("🧠 Creazione embeddings...")
	embeddings, err := kb.Embeddings.EmbedDocuments(context.Background(), chunks)
	if err != nil {
		return fmt.Errorf("errore nella creazione embeddings: %w", err)
	}

	// Salva chunks con embeddings
	kb.Chunks = make([]DocumentChunk, len(chunks))
	for i, chunk := range chunks {
		kb.Chunks[i] = DocumentChunk{
			Text:      chunk,
			Embedding: embeddings[i],
		}
	}

	fmt.Printf("✅ Knowledge base creata con %d chunks\n", len(kb.Chunks))
	return nil
}

// SaveToFile salva la knowledge base su file
func (kb *KnowledgeBase) SaveToFile(filename string) error {
	fmt.Printf("💾 Salvataggio knowledge base: %s\n", filename)

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(kb.Chunks)
}

// LoadFromFile carica la knowledge base da file
func (kb *KnowledgeBase) LoadFromFile(filename string) error {
	fmt.Printf("📂 Caricamento knowledge base: %s\n", filename)

	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	return decoder.Decode(&kb.Chunks)
}

// Search cerca i chunks più rilevanti per una query
func (kb *KnowledgeBase) Search(query string, topK int) ([]DocumentChunk, error) {
	// Crea embedding per la query
	queryEmbedding, err := kb.Embeddings.EmbedQuery(context.Background(), query)
	if err != nil {
		return nil, err
	}

	// Calcola similarità coseno con tutti i chunks
	results := make([]DocumentChunk, len(kb.Chunks))
	for i, chunk := range kb.Chunks {
		similarity := cosineSimilarity(queryEmbedding, chunk.Embedding)
		results[i] = DocumentChunk{
			Text:       chunk.Text,
			Embedding:  chunk.Embedding,
			Similarity: similarity,
		}
	}

	// Ordina per similarità decrescente
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Ritorna i top-k risultati
	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

// Chatbot struttura principale
type Chatbot struct {
	KB     *KnowledgeBase
	LLM    *ollama.LLM
	config *Config
}

// NewChatbot crea un nuovo chatbot
func NewChatbot(config *Config) (*Chatbot, error) {
	// Inizializza LLM Ollama
	llm, err := ollama.New(
		ollama.WithServerURL(config.OllamaURL),
		ollama.WithModel(config.Model),
	)
	if err != nil {
		return nil, fmt.Errorf("errore inizializzazione Ollama: %w", err)
	}

	// Inizializza Knowledge Base
	kb := NewKnowledgeBase(config)

	return &Chatbot{
		KB:     kb,
		LLM:    llm,
		config: config,
	}, nil
}

// LoadKnowledge carica la conoscenza da PDF o file salvato
func (c *Chatbot) LoadKnowledge(pdfPath string) error {
	dataFile := c.config.DataFile

	// Prova a caricare da file salvato
	if _, err := os.Stat(dataFile); err == nil {
		fmt.Println("🔄 Caricamento da file esistente...")
		if err := c.KB.LoadFromFile(dataFile); err == nil {
			fmt.Println("✅ Knowledge base caricata da file")
			return nil
		}
		fmt.Println("⚠️  Errore nel caricamento file, riprocesso PDF...")
	}

	// Carica da PDF
	if err := c.KB.LoadFromPDF(pdfPath); err != nil {
		return err
	}

	// Salva per uso futuro
	if err := c.KB.SaveToFile(dataFile); err != nil {
		fmt.Printf("⚠️  Errore nel salvataggio: %v\n", err)
	}

	return nil
}

// Ask fa una domanda al chatbot
func (c *Chatbot) Ask(query string) (string, error) {
	// Cerca chunks rilevanti
	relevantChunks, err := c.KB.Search(query, c.config.TopK)
	if err != nil {
		return "", fmt.Errorf("errore nella ricerca: %w", err)
	}

	// Verifica se ci sono risultati rilevanti
	if len(relevantChunks) == 0 || relevantChunks[0].Similarity < 0.3 {
		return "Non ho trovato informazioni rilevanti nel manuale per rispondere alla tua domanda. Prova a riformulare o usa termini più specifici.", nil
	}

	// Costruisci il contesto
	var contextBuilder strings.Builder
	contextBuilder.WriteString("INFORMAZIONI DAL MANUALE:\n\n")

	for i, chunk := range relevantChunks {
		if chunk.Similarity > 0.2 { // Soglia minima
			contextBuilder.WriteString(fmt.Sprintf("SEZIONE %d (similarità: %.2f):\n%s\n\n",
				i+1, chunk.Similarity, chunk.Text))
		}
	}

	// Crea il prompt
	prompt := fmt.Sprintf(`Sei un assistente tecnico specializzato. Rispondi alla domanda basandoti ESCLUSIVAMENTE sulle informazioni fornite dal manuale.

%s

DOMANDA: %s

ISTRUZIONI:
- Rispondi solo usando le informazioni del manuale fornite sopra
- Se le informazioni non sono sufficienti, dillo chiaramente
- Mantieni la risposta concisa ma completa
- Rispondi sempre in italiano

RISPOSTA:`, contextBuilder.String(), query)

	// Genera risposta con Ollama
	ctx := context.Background()
	response, err := c.LLM.Call(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("errore nella generazione risposta: %w", err)
	}

	return strings.TrimSpace(response), nil
}

// Funzioni di utilità

// extractTextFromPDF estrae testo da un file PDF
func extractTextFromPDF(pdfPath string) (string, error) {
	file, reader, err := pdf.Open(pdfPath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	var textBuilder strings.Builder

	for i := 1; i <= reader.NumPage(); i++ {
		page := reader.Page(i)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			continue
		}

		textBuilder.WriteString(text)
		textBuilder.WriteString("\n")
	}

	return cleanText(textBuilder.String()), nil
}

// cleanText pulisce e normalizza il testo
func cleanText(text string) string {
	// Rimuovi caratteri di controllo e normalizza spazi
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")

	// Rimuovi caratteri non stampabili
	re = regexp.MustCompile(`[^\p{L}\p{N}\p{P}\p{Z}]`)
	text = re.ReplaceAllString(text, "")

	return strings.TrimSpace(text)
}

// splitTextIntoChunks divide il testo in chunks
func splitTextIntoChunks(text string, chunkSize, overlap int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{}
	}

	var chunks []string
	start := 0

	for start < len(words) {
		end := start + chunkSize
		if end > len(words) {
			end = len(words)
		}

		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)

		if end == len(words) {
			break
		}

		start = end - overlap
		if start < 0 {
			start = 0
		}
	}

	return chunks
}

// cosineSimilarity calcola la similarità coseno tra due vettori
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}

// checkOllamaConnection verifica la connessione a Ollama
func checkOllamaConnection(url string) error {
	resp, err := http.Get(url + "/api/tags")
	if err != nil {
		return fmt.Errorf("impossibile connettersi a Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Ollama risponde con errore: %d", resp.StatusCode)
	}

	return nil
}

// checkModel verifica che il modello sia disponibile
func checkModel(url, model string) error {
	resp, err := http.Get(url + "/api/tags")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}

	for _, m := range result.Models {
		if strings.Contains(m.Name, model) {
			return nil
		}
	}

	return fmt.Errorf("modello %s non trovato", model)
}

// main function
func main() {
	fmt.Println("🤖 CHATBOT TECNICO OFFLINE")
	fmt.Println("=" + strings.Repeat("=", 40))

	// Configurazione
	config := &Config{
		OllamaURL:    "http://localhost:11434",
		Model:        "robomotic/gemma-2-2b-neogenesis-ita",
		ChunkSize:    300,
		ChunkOverlap: 50,
		TopK:         3,
		DataFile:     "knowledge_base.json",
	}

	// Verifica argomenti
	if len(os.Args) < 2 {
		fmt.Println("❌ Uso: go run main.go <percorso_manuale.pdf>")
		fmt.Println("Esempio: go run main.go manuale.pdf")
		os.Exit(1)
	}

	pdfPath := os.Args[1]

	// Verifica file PDF
	if _, err := os.Stat(pdfPath); os.IsNotExist(err) {
		fmt.Printf("❌ File PDF non trovato: %s\n", pdfPath)
		os.Exit(1)
	}

	// Verifica connessione Ollama
	fmt.Println("🔍 Verifica connessione Ollama...")
	if err := checkOllamaConnection(config.OllamaURL); err != nil {
		fmt.Printf("❌ %v\n", err)
		fmt.Println("💡 Assicurati che Ollama sia in esecuzione: ollama serve")
		os.Exit(1)
	}

	// Verifica modello
	fmt.Printf("🔍 Verifica modello %s...\n", config.Model)
	if err := checkModel(config.OllamaURL, config.Model); err != nil {
		fmt.Printf("❌ %v\n", err)
		fmt.Printf("💡 Scarica il modello: ollama pull %s\n", config.Model)
		os.Exit(1)
	}

	// Verifica modello embedding
	fmt.Println("🔍 Verifica modello embedding...")
	if err := checkModel(config.OllamaURL, "nomic-embed-text"); err != nil {
		fmt.Printf("❌ Modello embedding non trovato\n")
		fmt.Println("💡 Scarica il modello: ollama pull nomic-embed-text")
		os.Exit(1)
	}

	fmt.Println("✅ Ollama configurato correttamente")

	// Inizializza chatbot
	fmt.Println("🚀 Inizializzazione chatbot...")
	chatbot, err := NewChatbot(config)
	if err != nil {
		fmt.Printf("❌ Errore inizializzazione: %v\n", err)
		os.Exit(1)
	}

	// Carica conoscenza
	fmt.Println("📚 Caricamento conoscenza...")
	if err := chatbot.LoadKnowledge(pdfPath); err != nil {
		fmt.Printf("❌ Errore caricamento: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ Chatbot pronto!")
	fmt.Println("\n💬 Puoi iniziare a fare domande. Digita 'exit' per uscire.")
	fmt.Println("=" + strings.Repeat("=", 50))

	// Loop interattivo
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n🤔 La tua domanda: ")

		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())

		if query == "" {
			continue
		}

		if strings.ToLower(query) == "exit" {
			fmt.Println("👋 Arrivederci!")
			break
		}

		fmt.Println("\n🧠 Elaborazione...")
		start := time.Now()

		response, err := chatbot.Ask(query)
		if err != nil {
			fmt.Printf("❌ Errore: %v\n", err)
			continue
		}

		elapsed := time.Since(start)

		fmt.Printf("\n🤖 Risposta (tempo: %.2fs):\n", elapsed.Seconds())
		fmt.Println(strings.Repeat("-", 50))
		fmt.Println(response)
		fmt.Println(strings.Repeat("-", 50))
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Errore lettura input: %v", err)
	}
}
