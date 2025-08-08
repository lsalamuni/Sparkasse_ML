#######################################################################################
#                                                                                     #
#                                1. PACKAGES                                          #
#                                                                                     #
#######################################################################################

pacotes <- c("dplyr", "knitr", "tinytex", "readxl", "tidyr", "ggplot2",
             "tidytext", "RColorBrewer", "wordcloud", "SnowballC", "tm")

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
  select(-`n.y`) %>%
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
       x = NULL, # Remove x-axis label
       y = "Count") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "grey98", color = NA),
        plot.background = element_rect(fill = "white", color = "black", size = 1),
        panel.grid.major.y = element_blank(), # Remove vertical grid (horizontal in coord_flip)
        panel.grid.minor = element_blank(), # Remove minor grid
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
#                                  5. PARSING                                         #
#                                                                                     #
#######################################################################################

text <- df %>%
  select(inquiry) %>%
  rename(string = "inquiry") %>%
  as_tibble()

head(text)

tokens <- text %>%
  unnest_tokens(word, string) %>%
  arrange(word)

head(tokens)

stopwords <- tibble(word = readLines("german_stopwords_full.txt", encoding = "UTF-8"),
                    stringsAsFactors = FALSE) %>%
  select(word)

head(stopwords)

tokens <- tokens %>%
  anti_join(stopwords)

head(tokens)
summary(tokens)

# Create non-stemmed version (original tokens after stopword removal)
tokens_non_stemmed <- tokens

# Create stemmed version
tokens_stemmed <- tokens %>%
  mutate(word = wordStem(word, language = "german"))

# Count occurrences for both versions
ocurrences_non_stemmed <- tokens_non_stemmed %>%
  count(word, sort = TRUE)

ocurrences_stemmed <- tokens_stemmed %>%
  count(word, sort = TRUE)

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

# Apply custom stopwords and length filtering to both
ocurrences_non_stemmed <- ocurrences_non_stemmed %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2)

ocurrences_stemmed <- ocurrences_stemmed %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2)

# Visualization for non-stemmed tokens
ocurrences_non_stemmed %>%
  filter(n >= 40) %>%
  ggplot() +
  geom_col(aes(x = reorder(word, n),
               y = n), 
           fill = "#5A9FD4",
           width = 0.7) +
  coord_flip() +
  labs(title = "Non-Stemmed Token Occurrences",
       subtitle = "First filtering (n ≥ 40)",
       x = NULL,
       y = "Count",
       caption = "Note: Original tokens after filtering for stop words") +
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
                     limits = c(0, max(filter(ocurrences_non_stemmed, n >= 40)$n) * 1.1))

# Visualization for stemmed tokens
ocurrences_stemmed %>%
  filter(n >= 40) %>%
  ggplot() +
  geom_col(aes(x = reorder(word, n),
               y = n), 
           fill = "#E85D57",
           width = 0.7) +
  coord_flip() +
  labs(title = "Stemmed Token Occurrences",
       subtitle = "First filtering (n ≥ 40)",
       x = NULL,
       y = "Count",
       caption = "Note: Stemmed tokens after filtering for stop words") +
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
                     limits = c(0, max(filter(ocurrences_stemmed, n >= 40)$n) * 1.1))

# Second filtering for non-stemmed tokens
ocurrences_non_stemmed %>%
  filter(n >= 12) %>%
  ggplot() +
  geom_col(aes(x = reorder(word, n),
               y = n), 
           fill = "#5A9FD4",
           width = 0.7) +
  coord_flip() +
  labs(title = "Non-Stemmed Token Occurrences",
       subtitle = "Second filtering (n ≥ 12)",
       x = NULL,
       y = "Count",
       caption = "Note: Non-stemmed with additional stop words filtering") +
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
                     limits = c(0, max(filter(ocurrences_non_stemmed, n >= 12)$n) * 1.1))

# Second filtering for stemmed tokens
ocurrences_stemmed %>%
  filter(n >= 12) %>%
  ggplot() +
  geom_col(aes(x = reorder(word, n),
               y = n), 
           fill = "#E85D57",
           width = 0.7) +
  coord_flip() +
  labs(title = "Stemmed Token Occurrences",
       subtitle = "Second filtering (n ≥ 12)",
       x = NULL,
       y = "Count",
       caption = "Note: Stemmed with additional stop words filtering") +
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
                     limits = c(0, max(filter(ocurrences_stemmed, n >= 12)$n) * 1.1))

# Word clouds for comparison
set.seed(19072025)

# Non-stemmed word cloud
wordcloud(words = ocurrences_non_stemmed$word,
          freq = ocurrences_non_stemmed$n,
          min.freq = 5,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(8, "Set2"))

# Stemmed word cloud
wordcloud(words = ocurrences_stemmed$word,
          freq = ocurrences_stemmed$n,
          min.freq = 5,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))

#######################################################################################
#                                                                                     #
#                              6. TF-IDF TRANSFORMATION                              #
#                                                                                     #
#######################################################################################

# Prepare stemmed tokens for TF-IDF
# First, we need to add document IDs to track which email each token belongs to
text_with_id <- df %>%
  select(n, inquiry, sub2) %>%
  rename(document = n,
         text = inquiry,
         category = sub2)

# Create stemmed tokens with document IDs
tokens_for_tfidf <- text_with_id %>%
  unnest_tokens(word, text) %>%
  anti_join(stopwords) %>%
  mutate(word = wordStem(word, language = "german")) %>%
  filter(!word %in% tolower(custom_stopwords)) %>%
  filter(nchar(word) > 2)

head(tokens_for_tfidf)

# Calculate TF-IDF
tfidf_data <- tokens_for_tfidf %>%
  count(document, word, sort = TRUE) %>%
  bind_tf_idf(word, document, n)

head(tfidf_data %>% arrange(desc(tf_idf)))

# Look at highest TF-IDF terms
tfidf_data %>%
  arrange(desc(tf_idf)) %>%
  head(20)

# Visualize top TF-IDF terms
tfidf_data %>%
  arrange(desc(tf_idf)) %>%
  head(20) %>%
  ggplot(aes(x = reorder(word, tf_idf), y = tf_idf)) +
  geom_col(fill = "#E85D57", width = 0.7) +
  coord_flip() +
  labs(title = "Top 20 TF-IDF Terms",
       subtitle = "Most distinctive terms across documents",
       x = NULL,
       y = "TF-IDF Score") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "grey98", color = NA),
        plot.background = element_rect(fill = "white", color = "black", size = 1),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "grey85", size = 0.5),
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 5)),
        plot.subtitle = element_text(size = 12, color = "grey50", hjust = 0.5, margin = margin(b = 20)),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_text(size = 11, color = "grey30"),
        axis.title.x = element_text(size = 12, color = "grey30", margin = margin(t = 10)),
        plot.margin = margin(20, 20, 20, 20))

#######################################################################################
#                                                                                     #
#                            7. DOCUMENT-TERM MATRIX                                 #
#                                                                                     #
#######################################################################################

# Create document-term matrix for machine learning
# Filter to keep only terms that appear in at least 2 documents and at most 80% of documents
dtm_data <- tfidf_data %>%
  group_by(word) %>%
  mutate(doc_count = n()) %>%
  ungroup() %>%
  filter(doc_count >= 2 & doc_count <= nrow(text_with_id) * 0.8) %>%
  select(document, word, tf_idf)

# Create wide format document-term matrix
dtm_wide <- dtm_data %>%
  pivot_wider(names_from = word, 
              values_from = tf_idf, 
              values_fill = 0)

# Add category labels
ml_dataset <- dtm_wide %>%
  left_join(text_with_id %>% select(document, category), by = "document") %>%
  relocate(category, .after = document)

# Display dataset dimensions
cat("Dataset dimensions:", nrow(ml_dataset), "documents x", ncol(ml_dataset)-2, "features\n")

# Show category distribution
ml_dataset %>%
  count(category, sort = TRUE) %>%
  print()

# Save the processed dataset for machine learning
head(ml_dataset[,1:10])

# Summary statistics
cat("\nFeature matrix summary:\n")
feature_matrix <- ml_dataset %>% select(-document, -category)
cat("- Number of features:", ncol(feature_matrix), "\n")
cat("- Sparsity:", round(sum(feature_matrix == 0) / (nrow(feature_matrix) * ncol(feature_matrix)) * 100, 2), "%\n")
cat("- Non-zero elements:", sum(feature_matrix > 0), "\n")

#######################################################################################
#                                                                                     #
#                          8. FEATURE ANALYSIS BY CATEGORY                          #
#                                                                                     #
#######################################################################################

# Analyze top TF-IDF terms by category
category_tfidf <- tokens_for_tfidf %>%
  count(category, word, sort = TRUE) %>%
  bind_tf_idf(word, category, n) %>%
  arrange(desc(tf_idf))

# Top terms per category
top_terms_by_category <- category_tfidf %>%
  group_by(category) %>%
  slice_max(tf_idf, n = 5) %>%
  ungroup()

print("Top 5 TF-IDF terms by category:")
top_terms_by_category %>%
  select(category, word, tf_idf) %>%
  print()

# Visualize top terms by category
top_terms_by_category %>%
  ggplot(aes(x = reorder_within(word, tf_idf, category), y = tf_idf, fill = category)) +
  geom_col(show.legend = FALSE) +
  scale_x_reordered() +
  facet_wrap(~category, scales = "free", ncol = 3) +
  coord_flip() +
  labs(title = "Top TF-IDF Terms by Category",
       subtitle = "Most distinctive terms for each email category",
       x = NULL,
       y = "TF-IDF Score") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "grey98", color = NA),
        plot.background = element_rect(fill = "white", color = "black", size = 1),
        strip.background = element_rect(fill = "grey90", color = "black"),
        strip.text = element_text(size = 8, face = "bold"),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 5)),
        plot.subtitle = element_text(size = 11, color = "grey50", hjust = 0.5, margin = margin(b = 15)),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 7),
        panel.grid = element_blank())