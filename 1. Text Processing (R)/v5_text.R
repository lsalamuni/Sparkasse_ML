#######################################################################################
#                                                                                     #
#                                1. PACKAGES                                          #
#                                                                                     #
#######################################################################################

pacotes <- c("dplyr", "knitr", "tinytex", "readxl", "tidyr", "ggplot2",
             "tidytext", "RColorBrewer", "wordcloud", "SnowballC", "tm",
             "stringr", "scales", "writexl", "Matrix", "irlba")

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

# Category distribution plot
category_plot <- df %>%
  count(category) %>%
  ggplot() +
  geom_col(aes(x = reorder(category, n), y = n), 
           fill = main_color,
           width = 0.7) +
  coord_flip() +
  geom_hline(yintercept = 10, linetype = "dashed", color = secondary_color, size = 0.8) +
  geom_hline(yintercept = 20, linetype = "dashed", color = secondary_color, size = 0.8) +
  labs(title = "Analysis by Category",
       subtitle = "Total Occurrences - Highly Imbalanced Distribution",
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

print(category_plot)

# CLASS IMBALANCE ANALYSIS
class_distribution <- groups %>%
  arrange(desc(n)) %>%
  mutate(
    cumulative_n = cumsum(n),
    cumulative_pct = round(cumsum(np), 2),
    imbalance_ratio = round(max(n) / n, 2))

print(class_distribution)

# Imbalance metrics as tibble
imbalance_metrics <- tibble(
  metric = c("Max/Min class ratio", "Gini coefficient", "Classes with <10% of data"),
  value = c(
    round(max(class_distribution$n) / min(class_distribution$n), 2),
    round(2 * sum(rank(class_distribution$n) * class_distribution$n) / 
          (length(class_distribution$n) * sum(class_distribution$n)) - 
          (length(class_distribution$n) + 1) / length(class_distribution$n), 3),
    sum(class_distribution$np < 10)),
  description = c("Ratio between largest and smallest class", 
                  "Measure of distribution inequality (0=perfect equality, 1=perfect inequality)",
                  paste("Out of", nrow(class_distribution), "total classes")))

print(imbalance_metrics)

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
    avg_word_length = char_count / pmax(word_count, 1),  # Avoid division by zero
    
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
    contains_iban = as.numeric(str_detect(inquiry, "[A-Z]{2}\\d{2}[A-Z0-9]+")),
    contains_amount = as.numeric(str_detect(inquiry, "\\d+[,.]?\\d*\\s*(€|EUR|Euro)")),
    
    # Banking domain indicators
    contains_account = as.numeric(str_detect(inquiry, regex("konto|account", ignore_case = TRUE))),
    contains_credit = as.numeric(str_detect(inquiry, regex("kredit|darlehen|finanzierung", ignore_case = TRUE))),
    contains_transfer = as.numeric(str_detect(inquiry, regex("überweisung|transfer|zahlung", ignore_case = TRUE))),
    contains_card = as.numeric(str_detect(inquiry, regex("karte|card|kreditkarte|ec-karte|girocard", ignore_case = TRUE))),
    
    # Urgency indicators
    is_urgent = as.numeric(str_detect(inquiry, regex("dringend|sofort|eilig|urgent|wichtig|asap", ignore_case = TRUE))),
    
    # Formality indicators
    has_greeting = as.numeric(str_detect(inquiry, regex("^(sehr geehrte|liebe|hallo|guten)", ignore_case = TRUE))),
    has_closing = as.numeric(str_detect(inquiry, regex("(mit freundlichen grüßen|mfg|lg|vg|beste grüße)$", ignore_case = TRUE)))
  ) %>%
  select(n, char_count:has_closing)

# Summary of structural features
email_features %>%
  select(-n) %>%
  summary() %>%
  print()

# Visualize structural features by category
email_features_with_cat <- email_features %>%
  left_join(df %>% select(n, category), by = "n")

# Average word count by category
email_features_with_cat %>%
  group_by(category) %>%
  summarise(avg_word_count = mean(word_count), .groups = "drop") %>%
  ggplot(aes(x = reorder(category, avg_word_count), y = avg_word_count)) +
  geom_col(fill = secondary_color,
           width = 0.7) +
  coord_flip() +
  labs(title = "Average Email Length by Category",
       subtitle = "Highly Imbalanced Distribution",
       x = NULL,
       y = "Average Word Count") +
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
        plot.margin = margin(20, 20, 20, 20))

#######################################################################################
#                                                                                     #
#                        6. UNIGRAM TOKENIZATION & TF-IDF                             #
#                                                                                     #
#######################################################################################

text <- df %>%
  select(inquiry) %>%
  rename(string = "inquiry") %>%
  as_tibble()

# Load stopwords with error handling
if(file.exists("german_stopwords_full.txt")) {
  stopwords <- tibble(word = readLines("german_stopwords_full.txt", encoding = "UTF-8"),
                      stringsAsFactors = FALSE) %>%
    select(word)
} else {
  # Fallback to basic German stopwords if file not found
  stopwords <- tibble(word = stopwords("de"))
}

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
  anti_join(stopwords, by = "word") %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2) %>%
  filter(!str_detect(word, "^\\d+$"))  # Remove pure numbers

# Create stemmed version
tokens_stemmed <- tokens %>%
  mutate(word = wordStem(word, language = "german"))

# Prepare for TF-IDF
text_with_id <- df %>%
  select(n, inquiry, category) %>%
  rename(document = n,
         text = inquiry)

# Create unigram TF-IDF
unigram_tfidf <- text_with_id %>%
  unnest_tokens(word, text) %>%
  anti_join(stopwords, by = "word") %>%
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

# Top bigrams and trigrams as tibbles
top_bigrams <- bigram_tfidf %>%
  arrange(desc(tf_idf)) %>%
  head(20) %>%
  mutate(rank = row_number()) %>%
  relocate(rank) %>%
  mutate(ngram_type = "Bigram")

top_trigrams <- trigram_tfidf %>%
  arrange(desc(tf_idf)) %>%
  head(20) %>%
  mutate(rank = row_number()) %>%
  relocate(rank) %>%
  mutate(ngram_type = "Trigram")

# Display top n-grams
ngram_summary <- bind_rows(top_bigrams, top_trigrams) %>%
  select(ngram_type, rank, word, document, tf_idf)

print(ngram_summary)

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
  customer_service = c("beratung", "termin", "filiale", "öffnungszeiten", "kontakt"))

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
    customer_service_keywords = str_count(tolower(text), paste(banking_keywords$customer_service, collapse = "|"))) %>%
  select(document, account_keywords:customer_service_keywords)

# Analyze banking features by category
banking_features_by_cat <- banking_features %>%
  left_join(text_with_id %>% select(document, category), by = "document") %>%
  group_by(category) %>%
  summarise(across(account_keywords:customer_service_keywords, mean), .groups = "drop")

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
       fill = "Avg Count") +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 5)),
        axis.line = element_blank(),
        axis.ticks = element_blank())

#######################################################################################
#                                                                                     #
#                          9. COMBINED FEATURE MATRIX                               #
#                                                                                     #
#######################################################################################

# FIXED: Create document-term matrices for each n-gram type
# Renamed function to avoid conflict with text2vec::create_dtm
build_feature_matrix <- function(tfidf_data, min_docs = 2, max_doc_prop = 0.8) {
  n_docs <- length(unique(text_with_id$document))
  
  # Validate inputs
  if(nrow(tfidf_data) == 0) {
    return(tibble(document = unique(text_with_id$document)))
  }
  
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

# Create DTMs with error handling
unigram_dtm <- build_feature_matrix(unigram_tfidf)
bigram_dtm <- build_feature_matrix(bigram_tfidf, min_docs = 2) %>%
  rename_with(~str_replace(.x, "word_", "bigram_"), -document)
trigram_dtm <- build_feature_matrix(trigram_tfidf, min_docs = 2) %>%
  rename_with(~str_replace(.x, "word_", "trigram_"), -document)

# Combine all features BEFORE dimensionality reduction
ml_dataset_full <- text_with_id %>%
  select(document, category) %>%
  left_join(email_features, by = c("document" = "n")) %>%
  left_join(banking_features, by = "document") %>%
  left_join(unigram_dtm, by = "document") %>%
  left_join(bigram_dtm, by = "document") %>%
  left_join(trigram_dtm, by = "document")

# Replace any remaining NA values with 0
ml_dataset_full <- ml_dataset_full %>%
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))

# Display dataset information as tibble
feature_summary_full <- tibble(
  description = c("Total documents", "Total features", "Email structure features", 
                  "Banking domain features", "Unigram features", 
                  "Bigram features", "Trigram features"),
  count = c(nrow(ml_dataset_full), 
            ncol(ml_dataset_full) - 2,
            ncol(email_features) - 1,
            ncol(banking_features) - 1,
            ncol(unigram_dtm) - 1,
            ncol(bigram_dtm) - 1,
            ncol(trigram_dtm) - 1)
)

print(feature_summary_full)

#######################################################################################
#                                                                                     #
#                      10. DIMENSIONALITY REDUCTION (PCA/SVD)                       #
#                                                                                     #
#######################################################################################

# Separate feature types for targeted reduction
structural_features <- ml_dataset_full %>%
  select(document, category, char_count:has_closing, account_keywords:customer_service_keywords)

text_features <- ml_dataset_full %>%
  select(document, starts_with(c("word_", "bigram_", "trigram_")))

# Apply SVD to text features (TF-IDF matrix) with error handling
if(ncol(text_features) > 1) {
  text_matrix <- text_features %>%
    select(-document) %>%
    as.matrix()
  
  # Remove zero-variance columns
  col_vars <- apply(text_matrix, 2, var, na.rm = TRUE)
  zero_var_cols <- which(col_vars == 0 | is.na(col_vars))
  if(length(zero_var_cols) > 0) {
    text_matrix <- text_matrix[, -zero_var_cols]
  }
  
  # Check if we still have features to work with
  if(ncol(text_matrix) > 0) {
    # Perform SVD with irlba (efficient for sparse matrices)
    n_components <- min(50, ncol(text_matrix) - 1, nrow(text_matrix) - 1)
    
    if(n_components > 0) {
      svd_result <- irlba(text_matrix, nv = n_components, nu = n_components)
      
      # Calculate explained variance
      singular_values <- svd_result$d
      explained_var <- (singular_values^2) / sum(singular_values^2)
      cumulative_var <- cumsum(explained_var)
      
      # Variance explained summary as tibble
      variance_summary <- tibble(
        component = 1:min(10, length(explained_var)),
        explained_variance_pct = round(explained_var[1:min(10, length(explained_var))] * 100, 2),
        cumulative_variance_pct = round(cumulative_var[1:min(10, length(explained_var))] * 100, 2)
      )
      
      print(variance_summary)
      
      # Select components explaining 95% variance or max 30 components
      n_components_keep <- which(cumulative_var >= 0.95)[1]
      if(is.na(n_components_keep) || n_components_keep > 30) {
        n_components_keep <- min(30, n_components)
      }
      
      # Dimensionality reduction summary
      reduction_summary <- tibble(
        metric = c("Components computed", "Components kept", "Variance explained (%)", 
                   "Original text features", "Reduction ratio (%)"),
        value = c(n_components, 
                  n_components_keep,
                  round(cumulative_var[n_components_keep] * 100, 2),
                  ncol(text_matrix),
                  round((1 - n_components_keep/ncol(text_matrix)) * 100, 2))
      )
      
      print(reduction_summary)
      
      # Create reduced text features
      if(n_components_keep == 1) {
        # Handle single component case - diag() returns scalar, not matrix
        text_features_reduced <- svd_result$u[, 1, drop = FALSE] * svd_result$d[1]
      } else {
        # Multiple components case - diag() works correctly
        text_features_reduced <- svd_result$u[, 1:n_components_keep] %*% diag(svd_result$d[1:n_components_keep])
      }
      colnames(text_features_reduced) <- paste0("SVD_", 1:n_components_keep)
      text_features_reduced <- as_tibble(text_features_reduced) %>%
        mutate(document = text_features$document) %>%
        relocate(document)
      
      # Visualization of explained variance
      plot(1:min(30, length(explained_var)), 
           cumulative_var[1:min(30, length(explained_var))], 
           type = "b", 
           main = "Cumulative Variance Explained by SVD Components",
           xlab = "Number of Components", 
           ylab = "Cumulative Variance Explained",
           col = main_color, 
           lwd = 2)
      abline(h = 0.95, lty = 2, col = secondary_color)
      abline(v = n_components_keep, lty = 2, col = secondary_color)
    } else {
      # No valid components, use empty tibble
      text_features_reduced <- tibble(document = text_features$document)
    }
  } else {
    # No text features after cleaning, use empty tibble
    text_features_reduced <- tibble(document = text_features$document)
  }
} else {
  # No text features to reduce
  text_features_reduced <- tibble(document = text_features$document)
}

#######################################################################################
#                                                                                     #
#                         11. SIMPLE EMBEDDINGS (ROBUST)                           #
#                                                                                     #
#######################################################################################

# FIXED: Simplified embedding approach using basic averaging
create_simple_embeddings <- function(texts, n_dims = 30) {
  # Create a simple TF-IDF based embedding
  # This is more robust than the complex text2vec approach
  
  # Tokenize and create vocabulary
  all_tokens <- texts %>%
    tolower() %>%
    str_replace_all("[^a-zäöüß ]", " ") %>%
    str_split("\\s+") %>%
    unlist() %>%
    .[nchar(.) > 2] %>%
    .[!. %in% tolower(custom_stopwords)]
  
  # Get most frequent words as our vocabulary
  vocab <- table(all_tokens) %>%
    sort(decreasing = TRUE) %>%
    head(min(500, length(.))) %>%
    names()
  
  if(length(vocab) == 0) {
    # Fallback: create random embeddings if no vocabulary
    embeddings <- matrix(rnorm(length(texts) * n_dims, 0, 0.1), 
                        nrow = length(texts), ncol = n_dims)
  } else {
    # Create simple embeddings based on word presence
    # This is a simplified but robust approach
    embeddings <- matrix(0, nrow = length(texts), ncol = n_dims)
    
    for(i in 1:length(texts)) {
      text_tokens <- texts[i] %>%
        tolower() %>%
        str_replace_all("[^a-zäöüß ]", " ") %>%
        str_split("\\s+") %>%
        unlist() %>%
        .[nchar(.) > 2]
      
      # Simple embedding: use hash of words to create consistent features
      for(j in 1:n_dims) {
        hash_values <- sapply(text_tokens, function(w) {
          if(w %in% vocab) {
            # Simple hash function
            hash_val <- sum(utf8ToInt(w)) %% 1000
            sin(hash_val * j / 100)
          } else {
            0
          }
        })
        embeddings[i, j] <- mean(hash_values, na.rm = TRUE)
      }
    }
  }
  
  # Normalize embeddings
  embeddings <- apply(embeddings, 2, function(x) (x - mean(x)) / (sd(x) + 1e-8))
  
  colnames(embeddings) <- paste0("Embed_", 1:n_dims)
  return(as_tibble(embeddings))
}

# Create embeddings
embedding_features <- create_simple_embeddings(df$inquiry, n_dims = 30)
embedding_features$document <- 1:nrow(df)
embedding_features <- embedding_features %>% relocate(document)

# Embedding summary
embedding_summary <- tibble(
  description = "Simple hash-based embedding features created",
  dimensions = ncol(embedding_features) - 1,
  method = "Hash-based word averaging (robust implementation)"
)

print(embedding_summary)

#######################################################################################
#                                                                                     #
#                    12. FINAL INTEGRATED FEATURE MATRIX                            #
#                                                                                     #
#######################################################################################

# Combine all feature types
ml_dataset <- structural_features %>%
  left_join(text_features_reduced, by = "document") %>%
  left_join(embedding_features, by = "document")

# Replace any remaining NA values
ml_dataset <- ml_dataset %>%
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))

# Feature matrix characteristics
feature_summary_final <- tibble(
  description = c("Total documents", "Total features", "Email structure features",
                  "Banking domain features", "SVD components (from TF-IDF)",
                  "Embedding features"),
  count = c(nrow(ml_dataset),
            ncol(ml_dataset) - 2,
            sum(str_detect(names(ml_dataset), "char_count|word_count|avg_|question_|exclamation_|total_punct|sentence_|capital_|all_caps|contains_|is_|has_")),
            sum(str_detect(names(ml_dataset), "_keywords")),
            sum(str_detect(names(ml_dataset), "^SVD_")),
            sum(str_detect(names(ml_dataset), "^Embed_"))))

print(feature_summary_final)

# Category distribution
category_counts <- ml_dataset %>%
  count(category, sort = TRUE)

print(category_counts)

# Feature matrix sparsity analysis
feature_matrix <- ml_dataset %>% select(-document, -category)
matrix_characteristics <- tibble(
  metric = c("Matrix dimensions", "Total elements", "Zero elements", 
             "Non-zero elements", "Sparsity (%)"),
  value = c(paste(nrow(feature_matrix), "x", ncol(feature_matrix)),
            nrow(feature_matrix) * ncol(feature_matrix),
            sum(feature_matrix == 0),
            sum(feature_matrix != 0),
            round(sum(feature_matrix == 0) / (nrow(feature_matrix) * ncol(feature_matrix)) * 100, 2))
)

print(matrix_characteristics)

#######################################################################################
#                                                                                     #
#                  13. FEATURE ANALYSIS FOR IMBALANCED CLASSES                      #
#                                                                                     #
#######################################################################################

# Analyze feature distribution across minority vs majority classes
minority_classes <- class_distribution %>%
  filter(np < 10) %>%
  pull(category)

majority_classes <- class_distribution %>%
  filter(np >= 10) %>%
  pull(category)

# Compare feature means between minority and majority classes
feature_comparison <- ml_dataset %>%
  mutate(class_type = ifelse(category %in% minority_classes, "Minority", "Majority")) %>%
  group_by(class_type) %>%
  summarise(across(where(is.numeric), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(cols = -class_type, names_to = "feature", values_to = "mean_value") %>%
  pivot_wider(names_from = class_type, values_from = mean_value) %>%
  mutate(diff = Majority - Minority,
         ratio = Majority / (Minority + 0.001)) %>%
  arrange(desc(abs(diff)))

# Top discriminative features
top_discriminative_features <- feature_comparison %>%
  head(10) %>%
  mutate(
    description = case_when(
      str_detect(feature, "SVD_") ~ "SVD Component",
      str_detect(feature, "Embed_") ~ "Embedding Dimension",
      str_detect(feature, "_keywords") ~ "Banking Keyword Feature",
      TRUE ~ "Structural Feature")) %>%
  select(feature, description, Minority, Majority, diff, ratio)

print(top_discriminative_features)

#######################################################################################
#                                                                                     #
#                         14. SAVE PROCESSED DATA & REPORT                          #
#                                                                                     #
#######################################################################################

# Save the final ML dataset
write_xlsx(ml_dataset, "ml_dataset_v5_final.xlsx")

# Save the full feature set (before reduction) for comparison
write_xlsx(ml_dataset_full, "ml_dataset_v5_full.xlsx")

# Create comprehensive report
report <- list(
  dataset_info = list(
    n_documents = nrow(ml_dataset),
    n_features_final = ncol(ml_dataset) - 2,
    n_features_before_reduction = ncol(ml_dataset_full) - 2,
    n_categories = length(unique(ml_dataset$category)),
    category_distribution = table(ml_dataset$category)
  ),
  feature_breakdown = list(
    email_structure = sum(str_detect(names(ml_dataset), "char_count|word_count|avg_|question_|exclamation_|total_punct|sentence_|capital_|all_caps|contains_|is_|has_")),
    banking_domain = sum(str_detect(names(ml_dataset), "_keywords")),
    svd_components = ifelse(exists("n_components_keep"), n_components_keep, 0),
    embedding_dims = 30,
    total_after_reduction = ncol(ml_dataset) - 2
  ),
  processing_info = list(
    stopwords_used = nrow(stopwords),
    custom_stopwords_used = length(custom_stopwords),
    stemming = "German (SnowballC)",
    min_word_length = 3,
    min_document_frequency = 2,
    max_document_proportion = 0.8,
    dimensionality_reduction = "SVD on TF-IDF matrix (when applicable)",
    embeddings = "Hash-based word averaging (30 dimensions)"
  )
)

saveRDS(report, "text_processing_report_v5.rds")
