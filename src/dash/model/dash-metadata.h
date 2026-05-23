#ifndef DASH_METADATA_H
#define DASH_METADATA_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ns3
{

/**
 * Lightweight CSV index for DASH per-segment payload sizes.
 * Expected columns:
 *   - segment_index
 *   - size_bytes_250k, size_bytes_750k, size_bytes_1500k,
 *     size_bytes_3000k, size_bytes_6000k, size_bytes_12000k
 */
class DashMetadataIndex
{
  public:
    DashMetadataIndex();

    bool Load(const std::string& csvPath, std::string* errorMessage);
    bool IsLoaded() const;
    uint32_t GetSegmentCount() const;
    bool GetSegmentSizeBytes(uint32_t segmentId, uint32_t bitrateBps, uint32_t& sizeBytes) const;

  private:
    std::vector<std::unordered_map<uint32_t, uint32_t>> m_segmentSizeByRate;
    bool m_loaded;
};

} // namespace ns3

#endif /* DASH_METADATA_H */
