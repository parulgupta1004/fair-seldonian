Removed the `experiments` extra, which installed `ray`. Nothing in the package imports it; the study runner uses `concurrent.futures.ProcessPoolExecutor`. by @parulgupta1004
