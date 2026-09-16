"""Prefer current-policy corrections without changing packet trust or physics."""


def pop_recent_correction(exchange):
    # Select the newest item currently queued. The normal pop implementation
    # still performs unpacking/staleness checks and increments consumption.
    # Other queued items retain FIFO order; all raw experience files survive.
    # An upload arriving after selection is eligible on the next request.
    with exchange.lock:
        if exchange.queue:
            exchange.queue.rotate(1)
    return exchange.pop()
