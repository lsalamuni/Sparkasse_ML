#######################################################################################
#                                                                                     #
#                                1. PACKAGES                                          #
#                                                                                     #
#######################################################################################

pacotes <- c("dplyr", "knitr", "tinytex", "readxl", "tidyr", "ggplot2",
             "tidytext", "RColorBrewer", "wordcloud", "SnowballC", "tm",
             "stringr", "scales", "writexl")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){ 
  instalador <- pacotes[!pacotes %in% installed.packages()] 
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T)
} else {
  sapply(pacotes, require, character = T)
}

#######################################################################################
#                                                                                     #
#                                2. DATA SET                                          #
#                                                                                     #
#######################################################################################

df <- read_xlsx("Dataset.xlsx")

df <- df %>%
  as_tibble() %>%
  rename("n" = Nr.,
         "main" = Hauptkategorie,
         "sub1" = Unterkategorie1,
         "sub2" = Unterkategorie2,
         "inquiry" = Kundenanfrage,
         "forward" = Weiterleitung)

#######################################################################################
#                                                                                     #
#                                3. CATEGORIES                                        #
#                                                                                     #
#######################################################################################

main <- df %>%
  select(main) %>%
  unique()

sub <- df %>%
  select(sub1) %>%
  unique

sub2 <- df %>%
  group_by(sub2) %>%
  summarise(n = n())

seq <- LETTERS[1:12]
seq <- seq %>%
  as_tibble()

groups <- sub2 %>%
  arrange(sub2) %>%
  cbind(seq) %>%
  rename("id" = value) %>%
  relocate(id, .before = "sub2") %>%
  mutate(category = paste0(id, sep = " - ", sub2),
         np = round((n/sum(n))*100, 2)) %>%
  relocate(n, .after = "category") %>%
  as_tibble()

rm(sub2, seq)

df <- df %>%
  left_join(groups, by = "sub2") %>%
  relocate(category, .after = "sub2") %>%
  relocate(id, .before = "category") %>%
  select(-c(`n.y`, np)) %>%
  rename(n = `n.x`)

#######################################################################################
#                                                                                     #
#                                4. STATUS QUO                                        #
#                                                                                     #
#######################################################################################

main_color <- "#E85D57"
secondary_color <- "#5A9FD4" 

df %>%
  count(category) %>%
  ggplot() +
  geom_col(aes(x = reorder(category, n), y = n), 
           fill = main_color,
           width = 0.7) +
  coord_flip() +
  geom_hline(yintercept = 10, linetype = "dashed", color = secondary_color, size = 0.8) +
  geom_hline(yintercept = 20, linetype = "dashed", color = secondary_color, size = 0.8) +
  labs(title = "Analysis by Category",
       subtitle = "Total Ocurrences",
       x = NULL,
       y = "Count") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "grey98", color = NA),
        plot.background = element_rect(fill = "white", color = "black", size = 1),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "grey85", size = 0.5),
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 5)),
        plot.subtitle = element_text(size = 12, color = "grey50", hjust = 0.5, margin = margin(b = 20)),
        plot.caption = element_text(size = 10, color = "grey60", hjust = 0, margin = margin(t = 15)),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_text(size = 11, color = "grey30"),
        axis.title.x = element_text(size = 12, color = "grey30", margin = margin(t = 10)),
        plot.margin = margin(20, 20, 20, 20)) +
  scale_y_continuous(expand = c(0, 0),
                     limits = c(0, max(count(df, category)$n) * 1.1))

groups %>%
  arrange(desc(n))

#######################################################################################
#                                                                                     #
#                           5. EMAIL STRUCTURE FEATURES                               #
#                                                                                     #
#######################################################################################

# Extract structural features from emails
email_features <- df %>%
  mutate(
    # Basic length features
    char_count = nchar(inquiry),
    word_count = str_count(inquiry, "\\b\\w+\\b"),
    avg_word_length = char_count / word_count,
    
    # Punctuation features
    question_marks = str_count(inquiry, "\\?"),
    exclamation_marks = str_count(inquiry, "!"),
    total_punctuation = str_count(inquiry, "[[:punct:]]"),
    
    # Sentence structure
    sentence_count = str_count(inquiry, "[.!?]+"),
    avg_sentence_length = word_count / pmax(sentence_count, 1),
    
    # Capitalization features
    capital_words = str_count(inquiry, "\\b[A-Z][a-z]+\\b"),
    all_caps_words = str_count(inquiry, "\\b[A-Z]{2,}\\b"),
    
    # Numeric patterns
    contains_numbers = str_count(inquiry, "\\d+"),
    contains_iban = str_detect(inquiry, "[A-Z]{2}\\d{2}[A-Z0-9]+"),
    contains_amount = str_detect(inquiry, "\\d+[,.]?\\d*\\s*(€|EUR|Euro)"),
    
    # Banking domain indicators
    contains_account = str_detect(inquiry, regex("konto|account", ignore_case = TRUE)),
    contains_credit = str_detect(inquiry, regex("kredit|darlehen|finanzierung", ignore_case = TRUE)),
    contains_transfer = str_detect(inquiry, regex("überweisung|transfer|zahlung", ignore_case = TRUE)),
    contains_card = str_detect(inquiry, regex("karte|card|kreditkarte|ec-karte|girocard", ignore_case = TRUE)),
    
    # Urgency indicators
    is_urgent = str_detect(inquiry, regex("dringend|sofort|eilig|urgent|wichtig|asap", ignore_case = TRUE)),
    
    # Formality indicators
    has_greeting = str_detect(inquiry, regex("^(sehr geehrte|liebe|hallo|guten)", ignore_case = TRUE)),
    has_closing = str_detect(inquiry, regex("(mit freundlichen grüßen|mfg|lg|vg|beste grüße)$", ignore_case = TRUE))
  ) %>%
  select(n, char_count:has_closing)

# Summary of structural features
email_features %>%
  select(-n) %>%
  summary() %>%
  print()

# Visualize structural features by category
email_features_with_cat <- email_features %>%
  left_join(df %>% select(n, sub2), by = "n")

# Average word count by category
email_features_with_cat %>%
  group_by(sub2) %>%
  summarise(avg_word_count = mean(word_count)) %>%
  ggplot(aes(x = reorder(sub2, avg_word_count), y = avg_word_count)) +
  geom_col(fill = secondary_color) +
  coord_flip() +
  labs(title = "Average Email Length by Category",
       x = NULL,
       y = "Average Word Count") +
  theme_minimal()

#######################################################################################
#                                                                                     #
#                        6. UNIGRAM TOKENIZATION & TF-IDF                           #
#                                                                                     #
#######################################################################################

text <- df %>%
  select(inquiry) %>%
  rename(string = "inquiry") %>%
  as_tibble()

# Load stopwords
stopwords <- tibble(word = readLines("german_stopwords_full.txt", encoding = "UTF-8"),
                    stringsAsFactors = FALSE) %>%
  select(word)

# Custom stopwords
custom_stopwords <- c(
  # for obvious reasons...
  "sparkasse",
  
  # greetings
  "grüße", "freundliche", "liebes", "team", "beste", "liebe", "lieber",
  "herzliche", "viele", "schöne", "mfg", "lg", "vg", "gruß", "grüßen",
  
  # treatment
  "herr", "frau", "name", "namen", "sehr", "geehrte", "geehrter",
  
  # names
  "schneider", "becker", "wagner", "schubert", "fischer", "wolf", "lena", 
  "paul", "laura", "jonas", "müller", "schmidt", "meyer", "weber", "max",
  "julia", "hofmann", "ben", "anna", "tim", "meier",
  
  # generic email words
  "tag", "tage", "mail", "email", "nachricht", "antwort", "frage",
  "bitte", "danke", "vielen", "dank", "mia", "gerne", "freundlichen")

# Unigram tokens
tokens <- text %>%
  unnest_tokens(word, string) %>%
  anti_join(stopwords) %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2) %>%
  filter(!str_detect(word, "^\\d+$"))  # Remove pure numbers

# Create stemmed version
tokens_stemmed <- tokens %>%
  mutate(word = wordStem(word, language = "german"))

# Prepare for TF-IDF
text_with_id <- df %>%
  select(n, inquiry, sub2) %>%
  rename(document = n,
         text = inquiry,
         category = sub2)

# Create unigram TF-IDF
unigram_tfidf <- text_with_id %>%
  unnest_tokens(word, text) %>%
  anti_join(stopwords) %>%
  mutate(word = wordStem(word, language = "german")) %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2) %>%
  filter(!str_detect(word, "^\\d+$")) %>%
  count(document, word, sort = TRUE) %>%
  bind_tf_idf(word, document, n)

#######################################################################################
#                                                                                     #
#                         7. N-GRAM EXTRACTION & TF-IDF                             #
#                                                                                     #
#######################################################################################

# Extract bigrams
bigram_tfidf <- text_with_id %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ", remove = FALSE) %>%
  filter(!word1 %in% stopwords$word & !word2 %in% stopwords$word) %>%
  filter(!word1 %in% tolower(custom_stopwords) & !word2 %in% tolower(custom_stopwords)) %>%
  select(-word1, -word2) %>%
  count(document, bigram, sort = TRUE) %>%
  bind_tf_idf(bigram, document, n) %>%
  rename(word = bigram)

# Extract trigrams
trigram_tfidf <- text_with_id %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ", remove = FALSE) %>%
  filter(!word1 %in% stopwords$word & !word2 %in% stopwords$word & !word3 %in% stopwords$word) %>%
  filter(!word1 %in% tolower(custom_stopwords) & 
         !word2 %in% tolower(custom_stopwords) & 
         !word3 %in% tolower(custom_stopwords)) %>%
  select(-word1, -word2, -word3) %>%
  count(document, trigram, sort = TRUE) %>%
  bind_tf_idf(trigram, document, n) %>%
  rename(word = trigram)

# Top bigrams and trigrams
cat("\nTop 20 Bigrams by TF-IDF:\n")
bigram_tfidf %>%
  arrange(desc(tf_idf)) %>%
  head(20) %>%
  print()

cat("\nTop 20 Trigrams by TF-IDF:\n")
trigram_tfidf %>%
  arrange(desc(tf_idf)) %>%
  head(20) %>%
  print()

#######################################################################################
#                                                                                     #
#                     8. BANKING DOMAIN-SPECIFIC FEATURES                           #
#                                                                                     #
#######################################################################################

# Banking-specific keyword detection with stemming
banking_keywords <- list(
  account_related = c("konto", "girokonto", "sparkonto", "tagesgeld", "festgeld"),
  credit_related = c("kredit", "darlehen", "finanzierung", "zinsen", "rate", "tilgung"),
  payment_related = c("überweisung", "zahlung", "lastschrift", "dauerauftrag", "sepa"),
  card_related = c("karte", "kreditkarte", "girocard", "pin", "tan", "sperrung"),
  investment_related = c("fonds", "aktie", "depot", "anlage", "rendite", "risiko"),
  insurance_related = c("versicherung", "schutz", "police", "schaden"),
  online_banking = c("online", "app", "banking", "login", "passwort", "sicherheit"),
  customer_service = c("beratung", "termin", "filiale", "öffnungszeiten", "kontakt")
)

# Stem banking keywords for matching
banking_keywords_stemmed <- lapply(banking_keywords, function(x) {
  wordStem(tolower(x), language = "german")
})

# Extract banking domain features
banking_features <- text_with_id %>%
  mutate(
    # Count occurrences of each keyword category
    account_keywords = str_count(tolower(text), paste(banking_keywords$account_related, collapse = "|")),
    credit_keywords = str_count(tolower(text), paste(banking_keywords$credit_related, collapse = "|")),
    payment_keywords = str_count(tolower(text), paste(banking_keywords$payment_related, collapse = "|")),
    card_keywords = str_count(tolower(text), paste(banking_keywords$card_related, collapse = "|")),
    investment_keywords = str_count(tolower(text), paste(banking_keywords$investment_related, collapse = "|")),
    insurance_keywords = str_count(tolower(text), paste(banking_keywords$insurance_related, collapse = "|")),
    online_banking_keywords = str_count(tolower(text), paste(banking_keywords$online_banking, collapse = "|")),
    customer_service_keywords = str_count(tolower(text), paste(banking_keywords$customer_service, collapse = "|"))
  ) %>%
  select(document, account_keywords:customer_service_keywords)

# Analyze banking features by category
banking_features_by_cat <- banking_features %>%
  left_join(text_with_id %>% select(document, category), by = "document") %>%
  group_by(category) %>%
  summarise(across(account_keywords:customer_service_keywords, mean))

# Visualize banking keyword distribution
banking_features_by_cat %>%
  pivot_longer(cols = -category, names_to = "keyword_type", values_to = "avg_count") %>%
  ggplot(aes(x = category, y = keyword_type, fill = avg_count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = main_color) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Banking Keywords Distribution by Category",
       x = "Category",
       y = "Keyword Type",
       fill = "Avg Count")

#######################################################################################
#                                                                                     #
#                          9. COMBINED FEATURE MATRIX                               #
#                                                                                     #
#######################################################################################

# Create document-term matrices for each n-gram type
# Filter to keep only terms that appear in at least 2 documents
create_dtm <- function(tfidf_data, min_docs = 2, max_doc_prop = 0.8) {
  n_docs <- length(unique(text_with_id$document))
  
  tfidf_data %>%
    group_by(word) %>%
    mutate(doc_count = n()) %>%
    ungroup() %>%
    filter(doc_count >= min_docs & doc_count <= n_docs * max_doc_prop) %>%
    select(document, word, tf_idf) %>%
    pivot_wider(names_from = word, 
                values_from = tf_idf, 
                values_fill = 0,
                names_prefix = "word_")
}

# Create DTMs
unigram_dtm <- create_dtm(unigram_tfidf)
bigram_dtm <- create_dtm(bigram_tfidf, min_docs = 2) %>%
  rename_with(~str_replace(.x, "word_", "bigram_"), -document)
trigram_dtm <- create_dtm(trigram_tfidf, min_docs = 2) %>%
  rename_with(~str_replace(.x, "word_", "trigram_"), -document)

# Combine all features
ml_dataset <- text_with_id %>%
  select(document, category) %>%
  left_join(email_features, by = c("document" = "n")) %>%
  left_join(banking_features, by = "document") %>%
  left_join(unigram_dtm, by = "document") %>%
  left_join(bigram_dtm, by = "document") %>%
  left_join(trigram_dtm, by = "document")

# Display dataset information
cat("\n=== FINAL ML DATASET SUMMARY ===\n")
cat("Total documents:", nrow(ml_dataset), "\n")
cat("Total features:", ncol(ml_dataset) - 2, "\n")  # Minus document and category columns
cat("\nFeature breakdown:\n")
cat("- Email structure features:", ncol(email_features) - 1, "\n")
cat("- Banking domain features:", ncol(banking_features) - 1, "\n")
cat("- Unigram features:", ncol(unigram_dtm) - 1, "\n")
cat("- Bigram features:", ncol(bigram_dtm) - 1, "\n")
cat("- Trigram features:", ncol(trigram_dtm) - 1, "\n")

# Category distribution
ml_dataset %>%
  count(category, sort = TRUE) %>%
  print()

# Feature matrix characteristics
feature_matrix <- ml_dataset %>% select(-document, -category)
cat("\nFeature matrix characteristics:\n")
cat("- Dimensions:", nrow(feature_matrix), "x", ncol(feature_matrix), "\n")
cat("- Sparsity:", round(sum(feature_matrix == 0) / (nrow(feature_matrix) * ncol(feature_matrix)) * 100, 2), "%\n")
cat("- Non-zero elements:", sum(feature_matrix > 0), "\n")

#######################################################################################
#                                                                                     #
#                         10. FEATURE IMPORTANCE ANALYSIS                           #
#                                                                                     #
#######################################################################################

# Analyze most important features per category
# Using mean TF-IDF values for text features and mean values for structural features

# Text features importance by category
text_features_importance <- ml_dataset %>%
  select(category, starts_with(c("word_", "bigram_", "trigram_"))) %>%
  group_by(category) %>%
  summarise(across(everything(), mean)) %>%
  pivot_longer(cols = -category, names_to = "feature", values_to = "mean_value") %>%
  group_by(category) %>%
  slice_max(mean_value, n = 10) %>%
  ungroup()

# Visualize top features per category
text_features_importance %>%
  ggplot(aes(x = reorder_within(feature, mean_value, category), y = mean_value)) +
  geom_col(fill = main_color) +
  scale_x_reordered() +
  facet_wrap(~category, scales = "free", ncol = 3) +
  coord_flip() +
  labs(title = "Top Text Features by Category",
       subtitle = "Based on mean TF-IDF values",
       x = NULL,
       y = "Mean TF-IDF") +
  theme_minimal() +
  theme(strip.text = element_text(size = 8, face = "bold"),
        axis.text.y = element_text(size = 6))

# Structural features importance
structural_importance <- ml_dataset %>%
  select(category, char_count:has_closing, account_keywords:customer_service_keywords) %>%
  group_by(category) %>%
  summarise(across(everything(), mean)) %>%
  pivot_longer(cols = -category, names_to = "feature", values_to = "mean_value")

# Heatmap of structural features
structural_importance %>%
  ggplot(aes(x = category, y = feature, fill = mean_value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 8)) +
  labs(title = "Structural Features Heatmap by Category",
       x = "Category",
       y = "Feature",
       fill = "Mean Value")

#######################################################################################
#                                                                                     #
#                            11. SAVE PROCESSED DATA                                #
#                                                                                     #
#######################################################################################

# Save the final ML dataset
write_xlsx(ml_dataset, "ml_dataset_final.xlsx")

# Create a summary report
report <- list(
  dataset_info = list(
    n_documents = nrow(ml_dataset),
    n_features = ncol(ml_dataset) - 2,
    n_categories = length(unique(ml_dataset$category)),
    category_distribution = table(ml_dataset$category)
  ),
  feature_breakdown = list(
    email_structure = ncol(email_features) - 1,
    banking_domain = ncol(banking_features) - 1,
    unigrams = ncol(unigram_dtm) - 1,
    bigrams = ncol(bigram_dtm) - 1,
    trigrams = ncol(trigram_dtm) - 1
  ),
  processing_info = list(
    stopwords_used = length(stopwords$word),
    custom_stopwords_used = length(custom_stopwords),
    stemming = "German (SnowballC)",
    min_word_length = 3,
    min_document_frequency = 2,
    max_document_proportion = 0.8
  )
)

saveRDS(report, "text_processing_report.rds")
