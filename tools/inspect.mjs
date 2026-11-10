import { asyncBufferFromFile, parquetMetadataAsync, parquetSchema } from 'hyparquet'

const path = process.argv[2]
const file = await asyncBufferFromFile(path)
const meta = await parquetMetadataAsync(file)
const schema = parquetSchema(meta)
console.log('rows:', Number(meta.num_rows))
console.log('created_by:', meta.created_by)
for (const col of schema.children) {
  const e = col.element
  console.log('field:', e.name, '| type:', e.type, '| converted:', e.converted_type ?? '', '| logical:', JSON.stringify(e.logical_type ?? ''))
}
