#!/usr/bin/Rscript

require(ggplot2)
require(scales)

NS_PER_SEC <- 1000 * 1000 * 1000

theme_set(theme_gray(base_size = 10))

GRAPH_WIDTH_CM <- 10
GRAPH_HEIGHT_CM <- 6

save_graph <- function(name, plot) {
  ggsave(paste0(name, '.pdf'), plot, width = GRAPH_WIDTH_CM, height = GRAPH_HEIGHT_CM, scale = 2.3, units = 'cm')
}

args <- commandArgs(trailingOnly = TRUE)

bench_id <- args[1]
commits <- args[2:length(args)]

data <- data.frame()

for (commit in commits) {
  file <- paste0('bench-', bench_id, '-', commit, '.csv')
  file_data <- read.csv(file)
  data <- rbind(data, data.frame(Commit=commit, Mode=file_data$mode, Kernel.mode=file_data$kernel, Version=file_data$version, Type=file_data$type, Precompute=file_data$precompute, t_cost=file_data$t_cost, m_cost=file_data$m_cost, lanes=file_data$lanes, ns_per_hash=file_data$ns_per_hash))
}

data$nph_normalized <- data$ns_per_hash / (data$m_cost * data$t_cost)

data$hashes_per_second <- NS_PER_SEC / data$ns_per_hash
data$hps_normalized <- data$m_cost * data$t_cost * data$hashes_per_second

make_plots_hps <- function(mode, kernel, type, precompute) {
  data_b <- data[data$m_cost >= 4096 & data$Mode == mode & data$Kernel.mode == kernel & data$Version == 'v1.3' & data$Type == paste0('Argon2', type) & data$Precompute == precompute,]
  
  if (length(data_b$hashes_per_second) != c(0)) {
    data_b_f_t_cost <- data_b$t_cost %in% c(1, 2, 4, 8, 16)
    data_b_f_m_cost <- data_b$m_cost %in% c(4096, 16384, 65536, 262144, 1048576)
    
    prefix <- paste0('plot-', bench_id, '-', mode, '-', kernel, '-argon2', type)
    if (precompute == 'yes') {
        prefix <- paste0(prefix, '-precompute')
    }
    
    save_graph(paste0(prefix, '-t_cost'),
               ggplot(data_b[data_b_f_m_cost,], aes(x=t_cost, y=hps_normalized, group=Commit, colour=Commit)) +
                 geom_line() +
                 scale_y_continuous(labels=comma) +
                 facet_grid(lanes~m_cost, labeller=label_both) +
                 xlab('t_cost') + ylab('Hashes per second (normalized)'))
    
    save_graph(paste0(prefix, '-m_cost'),
               ggplot(data_b[data_b_f_t_cost,], aes(x=m_cost, y=hps_normalized, group=Commit, colour=Commit)) +
                 geom_line() +
                 scale_x_log10() +
                 scale_y_continuous(labels=comma) +
                 facet_grid(t_cost~lanes, labeller=label_both) +
                 xlab('m_cost (log scale)') + ylab('Hashes per second (normalized)'))
    
    save_graph(paste0(prefix, '-lanes'),
               ggplot(data_b[data_b_f_t_cost & data_b_f_m_cost,], aes(x=lanes, y=hps_normalized, group=Commit, colour=Commit)) +
                 geom_line() +
                 scale_y_continuous(labels=comma) +
                 facet_grid(t_cost~m_cost, labeller=label_both) +
                 xlab('lanes') + ylab('Hashes per second (normalized)'))
  }
}

for (mode in c('opencl', 'cuda')) {
  for (kernel in c('by-segment', 'oneshot')) {
    make_plots_hps(mode, kernel, 'd', 'no')
    make_plots_hps(mode, kernel, 'i', 'no')
    make_plots_hps(mode, kernel, 'i', 'yes')
  }
}
