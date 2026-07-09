FROM golang:1.26-alpine AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /embedcache ./cmd/embedcache

FROM scratch
COPY --from=build /embedcache /embedcache
EXPOSE 8090
ENTRYPOINT ["/embedcache"]
CMD ["serve", "-listen", ":8090"]
