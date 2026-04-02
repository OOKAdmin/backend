const { SitemapStream, streamToPromise } = require("sitemap");
const fs = require("fs");

const BASE_URL = "https://ookcalculator.com/"; // Replace with your actual domain

// Define your React site pages
const pages = [
  "/",
  "/BeamProperties",
  "/PadEye",
  "/BeamDeflection",
  "/AboutUs",
  "/Policy",
];

// Function to generate sitemap
async function generateSitemap() {
  const sitemap = new SitemapStream({ hostname: BASE_URL });

  pages.forEach((page) => {
    sitemap.write({ url: page, changefreq: "weekly", priority: 0.8 });
  });

  sitemap.end();

  const sitemapData = await streamToPromise(sitemap);
  fs.writeFileSync("public/sitemap.xml", sitemapData.toString()); // Save it inside 'public' folder
}

generateSitemap().then(() => console.log("✅ Sitemap generated successfully!"));
