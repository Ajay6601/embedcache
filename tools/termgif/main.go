// termgif renders an animated terminal session to a GIF for the README and
// docs site — reproducible from code, no screen recording involved.
//
// It is a separate Go module so the main embedcache module stays
// dependency-free (this tool needs golang.org/x/image for font rendering).
//
// Usage: go run . -out ../../docs/assets/demo.gif
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

var outPath = flag.String("out", "demo.gif", "output gif path")

var (
	bg    = color.RGBA{0x0b, 0x0f, 0x14, 0xff}
	barBg = color.RGBA{0x11, 0x18, 0x23, 0xff}
	text  = color.RGBA{0xd7, 0xe0, 0xea, 0xff}
	dim   = color.RGBA{0x5c, 0x69, 0x75, 0xff}
	green = color.RGBA{0x34, 0xd3, 0x99, 0xff}
	mint  = color.RGBA{0x6e, 0xe7, 0xb7, 0xff}
	amber = color.RGBA{0xfb, 0xbf, 0x24, 0xff}
	cyan  = color.RGBA{0x67, 0xe8, 0xf9, 0xff}
	red   = color.RGBA{0xff, 0x5f, 0x57, 0xff}
	ylw   = color.RGBA{0xfe, 0xbc, 0x2e, 0xff}
	grn2  = color.RGBA{0x28, 0xc8, 0x40, 0xff}
)

var palette = color.Palette{bg, barBg, text, dim, green, mint, amber, cyan, red, ylw, grn2}

const (
	width   = 760
	height  = 460
	marginX = 22
	topY    = 58 // below the title bar
	lineH   = 19
	fps     = 10 // 100ms per frame
)

type span struct {
	text string
	col  color.RGBA
}
type row []span

func cmd(s string) row  { return row{{"$ ", green}, {s, text}} }
func out(s string) row  { return row{{s, text}} }
func dimr(s string) row { return row{{s, dim}} }

// the demo script: the waste report, the product's money shot
type step struct {
	r      row
	typing bool // reveal char by char
	pause  int  // extra hold frames after the row completes
}

func script() []step {
	return []step{
		{r: cmd("embedcache analyze requests.jsonl"), typing: true, pause: 6},
		{r: out("embedcache offline waste analysis")},
		{r: out("=================================")},
		{r: out("requests analyzed        20000")},
		{r: out("embedding items          20000")},
		{r: out("unique items             2052"), pause: 2},
		{r: row{{"duplicate items          ", text}, {"17948   (89.7% of all items)", amber}}, pause: 3},
		{r: out("estimated tokens         340217   (~$44.23)")},
		{r: row{{"estimated wasted tokens  ", text}, {"305090   (~$39.66)", amber}}, pause: 5},
		{r: out("")},
		{r: row{{">> 89.7% of this embedding spend was duplicate work", mint}}},
		{r: row{{">> an exact-match cache would have absorbed.", mint}}, pause: 6},
		{r: out("")},
		{r: out("top duplicated inputs:")},
		{r: row{{"  4831x  ", cyan}, {`tokens=7    "how do I reset my password"`, text}}},
		{r: row{{"  2210x  ", cyan}, {`tokens=4    "pricing plans"`, text}}},
		{r: row{{"  1187x  ", cyan}, {`tokens=9    "cancel my subscription"`, text}}, pause: 8},
		{r: out("")},
		{r: cmd("embedcache serve -upstream http://localhost:11434"), typing: true, pause: 4},
		{r: dimr("embedcache listening on :8090 -> ollama, remembering everything"), pause: 24},
	}
}

func main() {
	flag.Parse()
	face := loadFace()

	var frames []*image.Paletted
	var rows []row

	render := func(partial *row) {
		frames = append(frames, drawFrame(face, rows, partial))
	}

	for _, st := range script() {
		if st.typing {
			// reveal the command 3 chars per frame
			full := st.r[len(st.r)-1].text
			for n := 0; n <= len(full); n += 3 {
				if n > len(full) {
					n = len(full)
				}
				p := row{st.r[0], {full[:min(n, len(full))], text}}
				render(&p)
			}
		}
		rows = append(rows, st.r)
		render(nil)
		for i := 0; i < st.pause; i++ {
			render(nil)
		}
		// scroll if we run out of vertical space
		maxRows := (height - topY - 14) / lineH
		if len(rows) > maxRows {
			rows = rows[len(rows)-maxRows:]
		}
	}
	// final hold
	for i := 0; i < 20; i++ {
		render(nil)
	}

	g := &gif.GIF{LoopCount: 0}
	for i, f := range frames {
		delay := 100 / (1000 / fps / 10) // 10 = 100ms in GIF centiseconds
		_ = i
		g.Image = append(g.Image, f)
		g.Delay = append(g.Delay, delay)
	}

	fh, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer fh.Close()
	if err := gif.EncodeAll(fh, g); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s: %d frames, %dx%d\n", *outPath, len(frames), width, height)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func loadFace() font.Face {
	f, err := opentype.Parse(gomono.TTF)
	if err != nil {
		panic(err)
	}
	face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: 13, DPI: 72, Hinting: font.HintingFull})
	if err != nil {
		panic(err)
	}
	return face
}

func drawFrame(face font.Face, rows []row, partial *row) *image.Paletted {
	img := image.NewPaletted(image.Rect(0, 0, width, height), palette)
	fill(img, image.Rect(0, 0, width, height), bg)
	// title bar
	fill(img, image.Rect(0, 0, width, 34), barBg)
	circle(img, 20, 17, 6, red)
	circle(img, 40, 17, 6, ylw)
	circle(img, 60, 17, 6, grn2)
	drawText(img, face, 80, 22, "embedcache", dim)

	y := topY
	for _, r := range rows {
		x := marginX
		for _, s := range r {
			drawText(img, face, x, y, s.text, s.col)
			x += textWidth(face, s.text)
		}
		y += lineH
	}
	if partial != nil {
		x := marginX
		for _, s := range *partial {
			drawText(img, face, x, y, s.text, s.col)
			x += textWidth(face, s.text)
		}
		// cursor block
		fill(img, image.Rect(x+1, y-11, x+9, y+3), green)
	} else if len(rows) > 0 {
		fill(img, image.Rect(marginX+1, y-11, marginX+9, y+3), green)
	}
	return img
}

func fill(img *image.Paletted, r image.Rectangle, c color.RGBA) {
	idx := uint8(palette.Index(c))
	for yy := r.Min.Y; yy < r.Max.Y; yy++ {
		if yy < 0 || yy >= height {
			continue
		}
		for xx := r.Min.X; xx < r.Max.X; xx++ {
			if xx < 0 || xx >= width {
				continue
			}
			img.SetColorIndex(xx, yy, idx)
		}
	}
}

func circle(img *image.Paletted, cx, cy, r int, c color.RGBA) {
	for yy := -r; yy <= r; yy++ {
		for xx := -r; xx <= r; xx++ {
			if xx*xx+yy*yy <= r*r {
				img.SetColorIndex(cx+xx, cy+yy, uint8(palette.Index(c)))
			}
		}
	}
}

func drawText(img *image.Paletted, face font.Face, x, y int, s string, c color.RGBA) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: face,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
}

func textWidth(face font.Face, s string) int {
	d := &font.Drawer{Face: face}
	return d.MeasureString(s).Ceil()
}
